// Khronos glTF-Validator wrapper. Lives in scripts/ so node resolves
// gltf-validator from the project's node_modules; a script in a temp directory
// cannot, which fails as an opaque "not installed" message.
const v = require('gltf-validator');
const fs = require('fs');
v.validateBytes(new Uint8Array(fs.readFileSync(process.argv[2])))
  .then(r => console.log(JSON.stringify({
    e: r.issues.numErrors, w: r.issues.numWarnings,
    msgs: r.issues.messages.filter(m => m.severity < 2)
             .map(m => m.code + ' ' + (m.pointer || ''))})))
  .catch(e => console.log(JSON.stringify({e: -1, msgs: [String(e)]})));
