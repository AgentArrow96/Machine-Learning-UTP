// Run the official Playwright CLI with a file-sourced callback, preserving quotes/newlines.
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const root=path.resolve(__dirname,'..');
process.chdir(root);
const npx=path.join(root,'.cache/npm/_npx');
const cli=fs.existsSync(npx)?fs.readdirSync(npx).map(id=>path.join(npx,id,'node_modules/@playwright/cli/playwright-cli.js')).find(p=>fs.existsSync(p)):null;
if(!cli){console.error('Install the official CLI with a project-local cache: $env:npm_config_cache="$PWD/.cache/npm"; npx.cmd --yes --package @playwright/cli playwright-cli --help');process.exit(2);}
const output=path.join(root,'output/playwright');fs.mkdirSync(output,{recursive:true});
const env={...process.env,PLAYWRIGHT_MCP_OUTPUT_DIR:output,TEMP:path.join(root,'.tmp'),TMP:path.join(root,'.tmp')};
const code=fs.readFileSync(path.join(__dirname,'browser_checks.js'),'utf8');
const result=spawnSync(process.execPath,[cli,'-s=floodnet','run-code',code],{cwd:root,env,encoding:'utf8'});
fs.writeFileSync(path.join(output,'checks.log'),(result.stdout||'')+(result.stderr||''));
process.stdout.write(result.stdout||'');process.stderr.write(result.stderr||'');
process.exit(result.status||(/### Error/.test(result.stdout||'')?1:0));
