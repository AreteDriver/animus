(()=>{
const $=id=>document.getElementById(id);
const chat=$('chat'),msg=$('msg'),send=$('send'),bar=$('budget-bar'),blabel=$('budget-label'),qpanel=$('queue-panel'),qtoggle=$('queue-toggle'),qlist=$('queue-list'),qcount=$('queue-count');
const ESC={"&":"&amp;","<":"&lt;",">":"&gt;"};
const esc=s=>s.replace(/[&<>]/g,c=>ESC[c]);
const fmt=s=>esc(s).replace(/```([\s\S]*?)```/g,'<pre><code>$1</code></pre>').replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/\*([^*]+)\*/g,'<em>$1</em>').replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>').replace(/\n/g,'<br>');
const ICONS={ok:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>',run:'<svg class="spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>',err:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',wait:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>'};
let streaming=false;
function mk(t,c,n=''){const el=document.createElement(t);el.className=c;if(n)el.textContent=n;return el}
function addMsg(role,text){const wrap=mk('div','msg '+role);wrap.innerHTML=text?fmt(text):'<span class="empty">...</span>';chat.appendChild(wrap);chat.scrollTop=chat.scrollHeight;return wrap}
function updLast(el,text){el.innerHTML=fmt(text);chat.scrollTop=chat.scrollHeight}
async function postChat(body){const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!r.ok||!r.body){const j=await r.json().catch(()=>({}));throw new Error(j.error||j.detail||'Network error')}return r.body.getReader()}
async function stream(contentEl,payload){const reader=await postChat(payload);const dec=new TextDecoder();let buf='',text='';while(true){const{done,value}=await reader.read();if(done)break;buf+=dec.decode(value,{stream:true});const chunks=buf.split('\n\n');buf=chunks.pop();for(const chunk of chunks){const lines=chunk.split('\n');let ev='message',data=[];for(const ln of lines){if(ln.startsWith('event:'))ev=ln.slice(6).trim();else if(ln.startsWith('data:'))data.push(ln.slice(5).trim())}const d=data.join('\n');if(ev==='done'){streaming=false;send.disabled=false;return}if(ev==='error'){const j=JSON.parse(d||'{}');updLast(contentEl,(text?text+'\n':'')+'Error: '+(j.error||j.detail||'Unknown'));streaming=false;send.disabled=false;return}try{const j=JSON.parse(d||'{}');if(j.token!=null){text+=j.token;updLast(contentEl,text)}}catch(_){}}}streaming=false;send.disabled=false;chat.scrollTop=chat.scrollHeight}
function doSend(){const v=msg.value.trim();if(!v||streaming)return;msg.value='';msg.rows=1;streaming=true;send.disabled=true;const el=addMsg('user',v);addMsg('bot','');stream(el.nextElementSibling,{message:v})}
msg.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();doSend()}});
send.addEventListener('click',doSend);
qtoggle.addEventListener('click',()=>{const c=qpanel.classList.toggle('collapsed');qtoggle.setAttribute('aria-expanded',String(!c))});
async function loadBudget(){try{const b=await(await fetch('/api/budget')).json();bar.style.width=b.percent+'%';bar.className='budget-bar '+b.status;blabel.textContent=`${Math.round(b.used)} / ${b.total} ET`}catch(_){}}
async function loadQueue(){try{const items=await(await fetch('/api/queue')).json();qcount.textContent=String(items.length);qlist.innerHTML='';for(const it of items){const li=mk('li','queue-item');li.innerHTML=`<span class="icon ${it.status}">${ICONS[it.status]||ICONS.wait}</span><span>${esc(it.name||'Build')}</span>`;qlist.appendChild(li)}qlist.style.display=items.length?'block':'none'}catch(_){}}
loadBudget();loadQueue();
setInterval(loadBudget,5000);
setInterval(loadQueue,5000);
})();
