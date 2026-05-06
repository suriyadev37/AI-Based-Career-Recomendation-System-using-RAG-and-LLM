/* ============================================================
   CareerPath AI — Shared JavaScript
   Covers: graduate, jobseeker, school, uneducated form pages
   ============================================================ */

/* ── Generic dynamic rows (single text input) ───────────── */
function addDynamic(containerId, fieldName, placeholder) {
  var container = document.getElementById(containerId);
  var row = document.createElement('div');
  row.className = 'dynamic-row';
  row.innerHTML =
    '<input type="text" name="' + fieldName + '" placeholder="' + placeholder + '">' +
    '<button type="button" class="remove-btn" onclick="removeRow(this)">✕</button>';
  container.appendChild(row);
}

/* ── Subject rows (school page — text + number + remove) ── */
function addSubject() {
  var container = document.getElementById('subjects-container');
  var row = document.createElement('div');
  row.className = 'dynamic-row has-marks';
  row.innerHTML =
    '<input type="text" name="subjects" placeholder="Subject name" required>' +
    '<input type="number" name="marks" placeholder="Mark %" min="0" max="100" required>' +
    '<button type="button" class="remove-btn" onclick="removeRow(this)">✕</button>';
  container.appendChild(row);
}

/* ── Skill rows (graduate / jobseeker / school pages) ────── */
function addSkill(proficiencyOptions) {
  var defaults = proficiencyOptions || [
    { value: 'Beginner',     label: 'Beginner' },
    { value: 'Intermediate', label: 'Intermediate', selected: true },
    { value: 'Advanced',     label: 'Advanced' }
  ];
  var optionsHtml = defaults.map(function(o) {
    return '<option value="' + o.value + '"' + (o.selected ? ' selected' : '') + '>' + o.label + '</option>';
  }).join('');

  var container = document.getElementById('skills-container');
  var row = document.createElement('div');
  row.className = 'skill-row';
  row.innerHTML =
    '<input type="text" name="skills" placeholder="Skill" required>' +
    '<select name="' + (window.SKILL_LEVEL_NAME || 'skill_levels') + '" required>' + optionsHtml + '</select>' +
    '<button type="button" class="remove-btn" onclick="removeRow(this)">✕</button>';
  container.appendChild(row);
}

/* ── Remove any dynamic row ──────────────────────────────── */
function removeRow(btn) {
  btn.closest('.dynamic-row, .skill-row').remove();
}

/* ── Loading overlay helpers ─────────────────────────────── */
var _loMessages = [
  'Analyzing your profile\u2026',
  'Consulting the AI advisor\u2026',
  'Building your career roadmap\u2026',
  'Almost there\u2026'
];
var _loIndex = 0;
var _loTimer = null;

function _createOverlay() {
  if (document.getElementById('career-loading-overlay')) return;
  var el = document.createElement('div');
  el.id = 'career-loading-overlay';
  el.className = 'loading-overlay';
  el.innerHTML =
    '<div class="lo-glow"></div>' +
    '<div class="lo-spinner"></div>' +
    '<div class="lo-content">' +
      '<div class="lo-title">\uD83D\uDD2E AI is thinking\u2026</div>' +
      '<div class="lo-sub" id="lo-sub-text">' + _loMessages[0] + '</div>' +
      '<div class="lo-dots"><span></span><span></span><span></span></div>' +
    '</div>';
  document.body.appendChild(el);
}

function _startMessageCycle() {
  _loIndex = 0;
  _loTimer = setInterval(function() {
    _loIndex = (_loIndex + 1) % _loMessages.length;
    var el = document.getElementById('lo-sub-text');
    if (el) el.textContent = _loMessages[_loIndex];
  }, 2800);
}

function _showOverlay() {
  _createOverlay();
  var overlay = document.getElementById('career-loading-overlay');
  overlay.classList.add('active');
  document.body.style.overflow = 'hidden';
  _startMessageCycle();
}

function _hideOverlay() {
  var overlay = document.getElementById('career-loading-overlay');
  if (overlay) overlay.classList.remove('active');
  document.body.style.overflow = '';
  if (_loTimer) { clearInterval(_loTimer); _loTimer = null; }
}

/* ── Public showLoading — called by form onsubmit ────────── */
function showLoading() {
  var btn = document.querySelector('.submit-btn');
  if (btn) btn.disabled = true;
  _showOverlay();
}

/* ── FIX: bfcache — reset loader when user hits Back ────────
   The browser may restore a frozen page from the bfcache
   with the overlay still visible. pageshow fires on both
   initial load AND bfcache restore (event.persisted = true). */
window.addEventListener('pageshow', function(event) {
  if (event.persisted) {
    _hideOverlay();
    var btn = document.querySelector('.submit-btn');
    if (btn) btn.disabled = false;
  }
});
