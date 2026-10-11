import * as THREE from 'three';

// All positions and joint locations are synthetic. Defaults preserve the preview;
// the main demo can pass already-loaded data and hide whole person groups.
export async function prepareOccupants({pose, layout, visualStyle = 'preparation'} = {}) {
  async function readData(filename) {
    const response = await fetch(new URL(filename, import.meta.url));
    if (!response.ok) throw new Error(`${filename}: HTTP ${response.status}`);
    return response.json();
  }
  [pose, layout] = await Promise.all([
    pose ?? readData('./skeleton-14.json'),
    layout ?? readData('./scene-layout.json'),
  ]);
  const group = new THREE.Group();
  group.name = 'Synthetic preparation layout';
  const demoStyle = visualStyle === 'demo';
  const jointGeometry = new THREE.SphereGeometry(demoStyle ? 0.052 : 0.045,12,8);
  const boneGeometry = new THREE.CylinderGeometry(demoStyle ? 0.021 : 0.016,demoStyle ? 0.021 : 0.016,1,8);
  const personMaterial = new THREE.MeshBasicMaterial({color:0x55ffe2,depthTest:false,depthWrite:false});
  const routerMaterial = new THREE.MeshBasicMaterial({color:demoStyle ? 0x74d682 : 0xffc86a,depthTest:false,depthWrite:false});
  const axis = new THREE.Vector3(0,1,0);
  const peoplePoses = new Map();
  const neutralJoints = Object.fromEntries(pose.joints.map(j=>[j.id,[...j.position]]));
  const delta = new THREE.Vector3();

  // The meshes own each person's local pose. Updating one never mutates the
  // source pose, another person, or the body's world position and direction.
  function updatePose(personId, joints) {
    const current = peoplePoses.get(personId);
    if (!current) return false;
    for (const [id, dot] of current.joints) dot.position.fromArray(joints[id]);
    for (const {mesh, start, end} of current.bones) {
      const a = current.joints.get(start).position;
      const b = current.joints.get(end).position;
      delta.subVectors(b,a);
      const length = delta.length();
      mesh.position.copy(a).add(b).multiplyScalar(.5);
      mesh.scale.y = length;
      if (length > 0) mesh.quaternion.setFromUnitVectors(axis,delta.divideScalar(length));
    }
    return true;
  }

  function resetPose(personId) {
    if (personId !== undefined) return updatePose(personId,neutralJoints);
    for (const id of peoplePoses.keys()) updatePose(id,neutralJoints);
  }

  function label(text, color) {
    const canvas=document.createElement('canvas');canvas.width=demoStyle ? 256 : 128;canvas.height=demoStyle ? 112 : 56;
    const ctx=canvas.getContext('2d');
    if(demoStyle){
      ctx.fillStyle='#080d0a';ctx.fillRect(0,0,256,112);
      ctx.strokeStyle='#74d682';ctx.lineWidth=4;ctx.strokeRect(2,2,252,108);
      ctx.fillStyle='#74d682';ctx.fillRect(18,31,7,50);
      ctx.fillStyle='#f4fff6';ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.font='bold 88px system-ui';ctx.fillText(text,137,59);
    }else{
      ctx.fillStyle='#122128';ctx.fillRect(0,0,128,56);
      ctx.fillStyle=color;ctx.textAlign='center';ctx.font='bold 34px system-ui';ctx.fillText(text,64,40);
    }
    const texture=new THREE.CanvasTexture(canvas);
    if(demoStyle)texture.colorSpace=THREE.SRGBColorSpace;
    const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,depthTest:false,depthWrite:false,toneMapped:!demoStyle}));
    sprite.scale.set(demoStyle ? 0.82 : 0.56,demoStyle ? 0.35875 : 0.245,1);sprite.renderOrder=40;return sprite;
  }
  for(const person of layout.people){
    const body=new THREE.Group();body.name=person.id;body.position.fromArray(person.position);
    body.rotation.y=THREE.MathUtils.degToRad(person.yawDegrees);
    const current={joints:new Map(),bones:[]};peoplePoses.set(person.id,current);
    // Each demo person owns its materials so revealing one cannot fade another.
    const jointMaterial=demoStyle ? new THREE.MeshBasicMaterial({color:0xe8ffec,depthTest:false,depthWrite:false,toneMapped:false}) : personMaterial;
    const boneMaterial=demoStyle ? new THREE.MeshBasicMaterial({color:0x74d682,depthTest:false,depthWrite:false,toneMapped:false}) : personMaterial;
    for(const joint of pose.joints){
      const dot=new THREE.Mesh(jointGeometry,jointMaterial);dot.name=`${person.id}:${joint.id}`;
      dot.position.fromArray(joint.position);dot.userData.kind='skeleton-joint';dot.renderOrder=30;body.add(dot);
      current.joints.set(joint.id,dot);
    }
    for(const [start,end] of pose.bones){
      const bone=new THREE.Mesh(boneGeometry,boneMaterial);bone.name=`${person.id}:${start}-${end}`;
      bone.userData.kind='skeleton-bone';bone.renderOrder=29;body.add(bone);
      current.bones.push({mesh:bone,start,end});
    }
    resetPose(person.id);
    const tag=label(person.id,'#55ffe2');tag.position.set(.68,1.45,0);body.add(tag);group.add(body);
  }
  for(const router of layout.routers){
    const node=new THREE.Group();node.name=router.id;node.position.fromArray(router.position);
    const box=new THREE.Mesh(new THREE.BoxGeometry(.30,.13,.20),routerMaterial);box.position.y=.065;box.renderOrder=30;node.add(box);
    for(const x of [-.1,.1]){
      const antenna=new THREE.Mesh(new THREE.CylinderGeometry(.012,.012,.20,6),routerMaterial);antenna.position.set(x,.23,0);antenna.renderOrder=30;node.add(antenna);
    }
    const tag=label(router.id,'#ffc86a');tag.position.set(.5,.35,0);node.add(tag);group.add(node);
  }
  let joints=0;group.traverse(o=>{if(o.userData.kind==='skeleton-joint')joints++;});
  return {group,people:layout.people.length,routers:layout.routers.length,joints,updatePose,resetPose};
}
