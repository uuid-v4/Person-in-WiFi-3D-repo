/*
 * CSI recorder with BLE upload.
 *
 * Captures CSI from the injectors (TX_MAC_PREFIX + id 1..15) on a fixed channel into a PSRAM ring buffer
 * and uploads it on demand over BLE (GATT notifications). Capture pauses during an upload
 * so the single 2.4 GHz radio is not time-shared with BLE while recording.
 *
 * Upload stream on DATA (notify): "HDR" header, then chunks [u32 first_index][records...], then "END".
 *
 * Live USB stream (when a host has the USB-Serial-JTAG port open): every record is also written as
 * 0xC5 0x1A 'R' + csi_rec_t + 1-byte sum of the record bytes, interleaved with text log lines.
 */
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "esp_csi_gain_ctrl.h"
#include "freertos/queue.h"
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"

#include "host/ble_hs.h"
#include "host/util/util.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

#define CSI_CHANNEL      6
#define BUF_BYTES        (7 * 1024 * 1024 + 512 * 1024)
#define REC_CSI_BYTES    104
#define FMT_VERSION      2

static const char *TAG = "csi_ble";
/* injector n uses MAC 02:c5:1a:00:00:0n; records carry n in flags bits 4..7 */
static const uint8_t TX_MAC_PREFIX[5] = {0x02, 0xc5, 0x1a, 0x00, 0x00};

typedef struct __attribute__((packed)) {
    uint32_t t_us;
    uint32_t seq;
    int8_t rssi;
    uint8_t rate;
    uint8_t agc;
    int8_t fft;
    uint8_t sig_mode;
    uint8_t flags;
    int8_t csi[REC_CSI_BYTES];
} csi_rec_t;

typedef struct __attribute__((packed)) {
    char magic[3];
    uint8_t version;
    uint32_t count;
    uint16_t rec_size;
    uint64_t dev_now_us;
    uint32_t total;
    uint32_t overwritten;
    uint8_t channel;
} dump_hdr_t;

static csi_rec_t *s_buf;
static uint32_t s_cap, s_head, s_count, s_total, s_overwritten;
static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
static volatile bool s_capturing = true;
static volatile bool s_dumping = false;

static char s_name[16];
static uint8_t s_own_addr_type;
static uint16_t s_conn = BLE_HS_CONN_HANDLE_NONE;
static uint16_t s_mtu = 23;
static uint16_t s_data_handle, s_stat_handle;
static TaskHandle_t s_dump_task;
static QueueHandle_t s_stream_q;

/* c51a000N-0000-4c5a-9e00-5f1a00c5c5a1, little-endian */
#define C51A_UUID(n) BLE_UUID128_INIT(0xa1, 0xc5, 0xc5, 0x00, 0x1a, 0x5f, 0x00, 0x9e, \
                                      0x5a, 0x4c, 0x00, 0x00, (n), 0x00, 0x1a, 0xc5)
static const ble_uuid128_t SVC_UUID = C51A_UUID(0x01);
static const ble_uuid128_t CTRL_UUID = C51A_UUID(0x02);
static const ble_uuid128_t DATA_UUID = C51A_UUID(0x03);
static const ble_uuid128_t STAT_UUID = C51A_UUID(0x04);

static void csi_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!s_capturing || !info || !info->buf || info->len < 128 || memcmp(info->mac, TX_MAC_PREFIX, 5) ||
        info->mac[5] == 0 || info->mac[5] > 15) {
        return;
    }
    csi_rec_t r;
    r.t_us = (uint32_t)esp_timer_get_time();
    r.flags = (uint8_t)(info->mac[5] << 4);
    r.seq = info->rx_seq;
    if (info->payload) {
        int end = info->payload_len < 96 ? info->payload_len : 96;
        for (int i = 0; i + 10 <= end; i++) {
            if (!memcmp(info->payload + i, "CSI-TX", 6)) {
                memcpy(&r.seq, info->payload + i + 6, 4);
                r.flags |= 1;
                break;
            }
        }
    }
    r.rssi = info->rx_ctrl.rssi;
    r.rate = info->rx_ctrl.rate;
    r.sig_mode = info->rx_ctrl.sig_mode;
    uint8_t agc = 0;
    int8_t fft = 0;
    esp_csi_gain_ctrl_get_rx_gain(&info->rx_ctrl, &agc, &fft);
    r.agc = agc;
    r.fft = fft;
    /* data subcarriers 1..26 and 38..63 of the 64-subcarrier LLTF; each is [imag, real] int8 */
    memcpy(r.csi, info->buf + 2, 52);
    memcpy(r.csi + 52, info->buf + 76, 52);

    portENTER_CRITICAL(&s_mux);
    s_buf[s_head] = r;
    s_head = (s_head + 1) % s_cap;
    if (s_count < s_cap) {
        s_count++;
    } else {
        s_overwritten++;
    }
    s_total++;
    portEXIT_CRITICAL(&s_mux);
    if (s_stream_q) {
        xQueueSend(s_stream_q, &r, 0);
    }
}

static void stream_task(void *arg)
{
    static uint8_t frame[3 + sizeof(csi_rec_t) + 1] = {0xC5, 0x1A, 'R'};
    csi_rec_t r;
    while (true) {
        if (xQueueReceive(s_stream_q, &r, portMAX_DELAY) != pdTRUE || !usb_serial_jtag_is_connected()) {
            continue;
        }
        memcpy(frame + 3, &r, sizeof(r));
        uint8_t sum = 0;
        for (size_t i = 0; i < sizeof(r); i++) {
            sum += frame[3 + i];
        }
        frame[sizeof(frame) - 1] = sum;
        usb_serial_jtag_write_bytes(frame, sizeof(frame), 0);
    }
}

static void wifi_init(void)
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_channel(CSI_CHANNEL, WIFI_SECOND_CHAN_NONE));

    wifi_promiscuous_filter_t filter = {.filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT | WIFI_PROMIS_FILTER_MASK_DATA};
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous_filter(&filter));
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = false,
        .shift = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

static int status_str(char *s, size_t n)
{
    uint32_t count, total, over;
    portENTER_CRITICAL(&s_mux);
    count = s_count;
    total = s_total;
    over = s_overwritten;
    portEXIT_CRITICAL(&s_mux);
    return snprintf(s, n, "name=%s cap=%d dump=%d buffered=%lu/%lu total=%lu overwritten=%lu ch=%d mtu=%u",
                    s_name, s_capturing, s_dumping, (unsigned long)count, (unsigned long)s_cap,
                    (unsigned long)total, (unsigned long)over, CSI_CHANNEL, s_mtu);
}

static int notify(const void *data, uint16_t len)
{
    for (int attempt = 0; attempt < 2000; attempt++) {
        if (s_conn == BLE_HS_CONN_HANDLE_NONE) {
            return -1;
        }
        struct os_mbuf *om = ble_hs_mbuf_from_flat(data, len);
        if (om && ble_gatts_notify_custom(s_conn, s_data_handle, om) == 0) {
            return 0;
        }
        vTaskDelay(1); /* at least one tick: pdMS_TO_TICKS(5) is 0 at the default 100 Hz tick */
    }
    return -1;
}

static void dump_task(void *arg)
{
    static uint8_t chunk[512];
    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        s_dumping = true;
        bool was_capturing = s_capturing;
        s_capturing = false;
        esp_wifi_set_csi(false);
        esp_wifi_set_promiscuous(false);
        vTaskDelay(pdMS_TO_TICKS(20));

        uint32_t count, start;
        dump_hdr_t hdr = {.magic = {'H', 'D', 'R'}, .version = FMT_VERSION, .rec_size = sizeof(csi_rec_t),
                          .channel = CSI_CHANNEL};
        portENTER_CRITICAL(&s_mux);
        count = s_count;
        start = (s_head + s_cap - s_count) % s_cap;
        hdr.total = s_total;
        hdr.overwritten = s_overwritten;
        portEXIT_CRITICAL(&s_mux);
        hdr.count = count;
        hdr.dev_now_us = esp_timer_get_time();

        int64_t t0 = esp_timer_get_time();
        bool ok = notify(&hdr, sizeof(hdr)) == 0;
        uint16_t payload = (s_mtu > 3 ? s_mtu - 3 : 20);
        if (payload > sizeof(chunk)) {
            payload = sizeof(chunk);
        }
        uint32_t per = (payload - 4) / sizeof(csi_rec_t);
        if (per == 0) {
            per = 1;
        }
        for (uint32_t i = 0; ok && i < count; i += per) {
            uint32_t k = (count - i < per) ? count - i : per;
            memcpy(chunk, &i, 4);
            for (uint32_t j = 0; j < k; j++) {
                memcpy(chunk + 4 + j * sizeof(csi_rec_t), &s_buf[(start + i + j) % s_cap], sizeof(csi_rec_t));
            }
            ok = notify(chunk, 4 + k * sizeof(csi_rec_t)) == 0;
        }
        if (ok) {
            char end[16];
            int n = snprintf(end, sizeof(end), "END%lu", (unsigned long)count);
            notify(end, n);
            portENTER_CRITICAL(&s_mux);
            s_count = 0;
            portEXIT_CRITICAL(&s_mux);
        } else {
            notify("ERR", 3);
        }
        ESP_LOGI(TAG, "dump %s: %lu records in %lld ms (mtu %u, %lu rec/notify)", ok ? "ok" : "ABORTED",
                 (unsigned long)count, (esp_timer_get_time() - t0) / 1000, s_mtu, (unsigned long)per);

        esp_wifi_set_promiscuous(true);
        esp_wifi_set_csi(true);
        s_capturing = was_capturing;
        s_dumping = false;
    }
}

static void handle_cmd(const char *cmd)
{
    ESP_LOGI(TAG, "cmd: %s", cmd);
    if (!strncmp(cmd, "DUMP", 4)) {
        if (!s_dumping) {
            xTaskNotifyGive(s_dump_task);
        }
    } else if (!strncmp(cmd, "STOP", 4)) {
        s_capturing = false;
    } else if (!strncmp(cmd, "START", 5)) {
        s_capturing = true;
    } else if (!strncmp(cmd, "NAME=", 5) && strlen(cmd + 5) > 0 && strlen(cmd + 5) <= 10) {
        nvs_handle_t h;
        if (nvs_open("csi", NVS_READWRITE, &h) == ESP_OK) {
            nvs_set_str(h, "name", cmd + 5);
            nvs_commit(h);
            nvs_close(h);
            ESP_LOGI(TAG, "name set to %s, restarting", cmd + 5);
            vTaskDelay(pdMS_TO_TICKS(300));
            esp_restart();
        }
    } else if (!strncmp(cmd, "FAKE", 4)) {
        csi_rec_t r = {0};
        portENTER_CRITICAL(&s_mux);
        for (uint32_t i = 0; i < 5000; i++) {
            r.t_us = (uint32_t)esp_timer_get_time() + i * 10000;
            r.seq = s_total;
            r.csi[0] = (int8_t)i;
            s_buf[s_head] = r;
            s_head = (s_head + 1) % s_cap;
            if (s_count < s_cap) {
                s_count++;
            }
            s_total++;
        }
        portEXIT_CRITICAL(&s_mux);
    } else if (!strncmp(cmd, "CLEAR", 5)) {
        portENTER_CRITICAL(&s_mux);
        s_count = 0;
        portEXIT_CRITICAL(&s_mux);
    }
}

static int gatt_access(uint16_t conn, uint16_t attr, struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    int which = (int)(intptr_t)arg;
    if (which == 2 && ctxt->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
        char cmd[32] = {0};
        uint16_t n = 0;
        ble_hs_mbuf_to_flat(ctxt->om, cmd, sizeof(cmd) - 1, &n);
        cmd[n] = 0;
        handle_cmd(cmd);
        return 0;
    }
    if (which == 4 && ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR) {
        char s[160];
        int n = status_str(s, sizeof(s));
        return os_mbuf_append(ctxt->om, s, n) == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
    }
    return which == 3 ? 0 : BLE_ATT_ERR_UNLIKELY;
}

static const struct ble_gatt_svc_def s_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &SVC_UUID.u,
        .characteristics = (struct ble_gatt_chr_def[]) {
            {.uuid = &CTRL_UUID.u, .access_cb = gatt_access, .arg = (void *)2,
             .flags = BLE_GATT_CHR_F_WRITE | BLE_GATT_CHR_F_WRITE_NO_RSP},
            {.uuid = &DATA_UUID.u, .access_cb = gatt_access, .arg = (void *)3,
             .flags = BLE_GATT_CHR_F_NOTIFY, .val_handle = &s_data_handle},
            {.uuid = &STAT_UUID.u, .access_cb = gatt_access, .arg = (void *)4,
             .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY, .val_handle = &s_stat_handle},
            {0},
        },
    },
    {0},
};

static void adv_start(void);

static int gap_event(struct ble_gap_event *ev, void *arg)
{
    switch (ev->type) {
    case BLE_GAP_EVENT_CONNECT:
        if (ev->connect.status == 0) {
            s_conn = ev->connect.conn_handle;
            ble_gap_set_data_len(s_conn, 251, 2120);
            ble_gap_set_prefered_le_phy(s_conn, BLE_GAP_LE_PHY_2M_MASK, BLE_GAP_LE_PHY_2M_MASK,
                                        BLE_GAP_LE_PHY_CODED_ANY);
            struct ble_gap_upd_params upd = {.itvl_min = 6, .itvl_max = 12, .latency = 0,
                                             .supervision_timeout = 400};
            ble_gap_update_params(s_conn, &upd);
            ESP_LOGI(TAG, "connected");
        } else {
            adv_start();
        }
        break;
    case BLE_GAP_EVENT_DISCONNECT:
        s_conn = BLE_HS_CONN_HANDLE_NONE;
        s_mtu = 23;
        ESP_LOGI(TAG, "disconnected");
        adv_start();
        break;
    case BLE_GAP_EVENT_MTU:
        s_mtu = ev->mtu.value;
        ESP_LOGI(TAG, "mtu %u", s_mtu);
        break;
    case BLE_GAP_EVENT_PHY_UPDATE_COMPLETE:
        ESP_LOGI(TAG, "phy tx=%d rx=%d", ev->phy_updated.tx_phy, ev->phy_updated.rx_phy);
        break;
    case BLE_GAP_EVENT_CONN_UPDATE: {
        struct ble_gap_conn_desc desc;
        if (ble_gap_conn_find(s_conn, &desc) == 0) {
            ESP_LOGI(TAG, "conn interval %.2f ms", desc.conn_itvl * 1.25);
        }
        break;
    }
    case BLE_GAP_EVENT_ADV_COMPLETE:
        adv_start();
        break;
    default:
        break;
    }
    return 0;
}

static void adv_start(void)
{
    struct ble_hs_adv_fields fields = {0};
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.name = (uint8_t *)s_name;
    fields.name_len = strlen(s_name);
    fields.name_is_complete = 1;
    ble_gap_adv_set_fields(&fields);
    struct ble_gap_adv_params params = {0};
    params.conn_mode = BLE_GAP_CONN_MODE_UND;
    params.disc_mode = BLE_GAP_DISC_MODE_GEN;
    ble_gap_adv_start(s_own_addr_type, NULL, BLE_HS_FOREVER, &params, gap_event, NULL);
}

static void on_sync(void)
{
    ble_hs_util_ensure_addr(0);
    ble_hs_id_infer_auto(0, &s_own_addr_type);
    adv_start();
}

static void host_task(void *param)
{
    nimble_port_run();
    nimble_port_freertos_deinit();
}

static void ble_init(void)
{
    ESP_ERROR_CHECK(nimble_port_init());
    ble_hs_cfg.sync_cb = on_sync;
    ble_svc_gap_init();
    ble_svc_gatt_init();
    ESP_ERROR_CHECK(ble_gatts_count_cfg(s_svcs));
    ESP_ERROR_CHECK(ble_gatts_add_svcs(s_svcs));
    ble_svc_gap_device_name_set(s_name);
    nimble_port_freertos_init(host_task);
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    s_buf = heap_caps_malloc(BUF_BYTES, MALLOC_CAP_SPIRAM);
    if (!s_buf) {
        ESP_LOGE(TAG, "PSRAM buffer allocation failed");
        abort();
    }
    s_cap = BUF_BYTES / sizeof(csi_rec_t);

    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    snprintf(s_name, sizeof(s_name), "CSI-%02X%02X%02X", mac[3], mac[4], mac[5]);
    nvs_handle_t h;
    char stored[11] = {0};
    size_t stored_len = sizeof(stored);
    if (nvs_open("csi", NVS_READONLY, &h) == ESP_OK) {
        if (nvs_get_str(h, "name", stored, &stored_len) == ESP_OK && stored[0]) {
            snprintf(s_name, sizeof(s_name), "CSI-%s", stored);
        }
        nvs_close(h);
    }
    ESP_LOGI(TAG, "%s: buffer %lu records x %u bytes", s_name, (unsigned long)s_cap, (unsigned)sizeof(csi_rec_t));

    usb_serial_jtag_driver_config_t usj = {.tx_buffer_size = 32768, .rx_buffer_size = 256};
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usj));
    usb_serial_jtag_vfs_use_driver();
    s_stream_q = xQueueCreate(512, sizeof(csi_rec_t));
    xTaskCreate(stream_task, "stream", 4096, NULL, 4, NULL);

    wifi_init();
    xTaskCreate(dump_task, "dump", 4096, NULL, 5, &s_dump_task);
    ble_init();

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(2000));
        char s[160];
        status_str(s, sizeof(s));
        printf("STAT %s\n", s);
        if (s_conn != BLE_HS_CONN_HANDLE_NONE && !s_dumping) {
            ble_gatts_chr_updated(s_stat_handle);
        }
    }
}
