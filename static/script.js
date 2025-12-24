const $ = (id) => document.getElementById(id);

function escapeHtml(str){
  return str
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

function renderSpans(spans){
  const html = spans.map(s => {
    const label = s.label || "O";
    const text = escapeHtml(s.text);
    if(label === "O"){
      return `<span class="ent O">${text}</span>`;
    }
    return `<span class="ent ${label}">${text}<span class="tag">${label}</span></span>`;
  }).join(" ");
  $("renderBox").innerHTML = html || "<em>Chưa có kết quả.</em>";
}

function renderTable(tokens, tags){
  let rows = "";
  for(let i=0;i<tokens.length;i++){
    rows += `<tr><td>${i}</td><td>${escapeHtml(tokens[i])}</td><td>${escapeHtml(tags[i] || "")}</td></tr>`;
  }
  const html = `
    <table>
      <thead><tr><th>#</th><th>Token</th><th>Tag</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
  $("tableBox").innerHTML = html;
}

async function predict(){
  const text = $("inputText").value.trim();
  const model = $("modelSelect").value;

  if(!text){
    $("renderBox").innerHTML = "<em>Nhập text trước đã.</em>";
    $("tableBox").innerHTML = "";
    return;
  }

  $("btnPredict").disabled = true;
  $("btnPredict").textContent = "Running...";

  try{
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({text, model})
    });
    const data = await res.json();
    if(!data.ok){
      $("renderBox").innerHTML = `<span class="ent O">❌ ${escapeHtml(data.error || "Error")}</span>`;
      $("tableBox").innerHTML = "";
      return;
    }
    renderSpans(data.spans || []);
    renderTable(data.tokens || [], data.tags || []);
  }catch(err){
    $("renderBox").innerHTML = `<span class="ent O">❌ ${escapeHtml(String(err))}</span>`;
    $("tableBox").innerHTML = "";
  }finally{
    $("btnPredict").disabled = false;
    $("btnPredict").textContent = "Predict";
  }
}

$("btnPredict").addEventListener("click", predict);
$("btnClear").addEventListener("click", () => {
  $("inputText").value = "";
  $("renderBox").innerHTML = "";
  $("tableBox").innerHTML = "";
});
$("btnExample").addEventListener("click", () => {
  $("inputText").value = "Nguyễn Công Phát học tại Trường Đại học Công nghệ Thông tin ở TP.HCM .";
});

$("inputText").addEventListener("keydown", (e) => {
  if((e.ctrlKey || e.metaKey) && e.key === "Enter"){
    predict();
  }
});
function fmt(x){
  if(x === null || x === undefined) return "-";
  return Number(x).toFixed(4);
}

async function loadMetrics(){
  const model = $("modelSelect").value;
  $("metricsBox").innerHTML = `<div class="muted">Loading metrics...</div>`;
  try{
    const res = await fetch(`/api/metrics?model=${encodeURIComponent(model)}`);
    const data = await res.json();
    if(!data.ok){
      $("metricsBox").innerHTML = `<div class="muted">❌ ${escapeHtml(data.error || "Error")}</div>`;
      return;
    }
    const m = data.metrics;
    const v = m.valid || {};
    const t = m.test || {};

    $("metricsBox").innerHTML = `
      <div class="metricsTitle">${escapeHtml(m.name || model)}</div>

      <div class="metricsGrid">
        <div class="metricCard">
          <div class="metricHead">VALID</div>
          <div class="metricRow"><span>Token F1 (all)</span><b>${fmt(v.token_f1_all)}</b></div>
          <div class="metricRow"><span>Token F1 (non-O)</span><b>${fmt(v.token_f1_non_o)}</b></div>
          <div class="metricRow"><span>Span P/R/F1</span><b>${fmt(v.span_p)} / ${fmt(v.span_r)} / ${fmt(v.span_f1)}</b></div>
        </div>

        <div class="metricCard">
          <div class="metricHead">TEST</div>
          <div class="metricRow"><span>Token F1 (all)</span><b>${fmt(t.token_f1_all)}</b></div>
          <div class="metricRow"><span>Token F1 (non-O)</span><b>${fmt(t.token_f1_non_o)}</b></div>
          <div class="metricRow"><span>Span P/R/F1</span><b>${fmt(t.span_p)} / ${fmt(t.span_r)} / ${fmt(t.span_f1)}</b></div>
        </div>
      </div>
    `;
  }catch(err){
    $("metricsBox").innerHTML = `<div class="muted">❌ ${escapeHtml(String(err))}</div>`;
  }
}
$("modelSelect").addEventListener("change", () => {
  loadMetrics();
});

loadMetrics();