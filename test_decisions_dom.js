/*
 * Executes the decision page in a real DOM and CLICKS the buttons.
 *
 * Written after two failed attempts at this UI that every markup assertion
 * passed. First the buttons had no selected style at all; then the styles were
 * emitted after </style> and applied to nothing. Both times the HTML looked
 * perfect and the page was dead. Checking markup cannot catch either — only
 * executing the script and reading a computed style can.
 *
 *   npm install jsdom && node test_decisions_dom.js path/to/page.html
 */
const fs = require('fs');
let JSDOM;
try {
  ({ JSDOM } = require('jsdom'));
} catch (e) {
  console.log('jsdom not installed - skipping the DOM test.');
  console.log('  npm install jsdom   (then re-run)');
  process.exit(0);          // absent tooling is not a failure
}

const PAGE = process.argv[2] || '/tmp/page.html';
const html = fs.readFileSync(PAGE, 'utf8');
const dom = new JSDOM(html, { runScripts: 'dangerously', pretendToBeVisual: true });
const { window } = dom;
const doc = window.document;

let fails = 0;
function check(name, cond, detail) {
  if (cond) { console.log('  [ok ] ' + name + (detail ? '  ' + detail : '')); }
  else { console.log('  [FAIL] ' + name + '  ' + (detail || '')); fails++; }
}

const cards = doc.querySelectorAll('.card');
check('cards rendered', cards.length > 0, cards.length + ' cards');

const card = cards[0];
const btns = card.querySelectorAll('.btns button');
check('three buttons on the card', btns.length === 3);

const accept = [...btns].find(b => b.dataset.choice === 'accept');
const reject = [...btns].find(b => b.dataset.choice === 'reject');
const other  = [...btns].find(b => b.dataset.choice === 'other');
check('data-choice hooks present', !!(accept && reject && other));

// --- before the click ---
const bgBefore = window.getComputedStyle(accept).backgroundColor;
console.log('\n  background before click: ' + bgBefore);
check('card starts undecided', card.classList.contains('undecided'));

// --- click Accept ---
accept.click();

check('handler ran (class added)', accept.classList.contains('on-accept'),
      'classes: ' + accept.className);
const bgAfter = window.getComputedStyle(accept).backgroundColor;
console.log('  background after click : ' + bgAfter);
check('BUTTON COLOUR CHANGED', bgAfter !== bgBefore, bgBefore + ' -> ' + bgAfter);
check('it is the accept green', /26,\s*122,\s*74|#1a7a4a/i.test(bgAfter), bgAfter);
check('text turned white', /255,\s*255,\s*255/.test(window.getComputedStyle(accept).color));
check('siblings dimmed', reject.classList.contains('dim') && other.classList.contains('dim'));
check('card marked accepted', card.classList.contains('accepted') &&
                              !card.classList.contains('undecided'));
const chip = doc.getElementById('st_' + card.id.replace('it_',''));
check('status chip updated', /accepted/.test(chip.textContent), JSON.stringify(chip.textContent));
check('chip styled ok', chip.className.includes('ok'), chip.className);

// --- switch to Reject: the fill must move ---
reject.click();
check('switching moves the fill',
      reject.classList.contains('on-reject') && !accept.classList.contains('on-accept'));
const rbg = window.getComputedStyle(reject).backgroundColor;
check('reject is red', /180,\s*65,\s*60/.test(rbg), rbg);
check('reason box revealed', doc.getElementById('why_' + card.id.replace('it_','')).style.display === 'block');

// --- the tally reacts ---
const tally = doc.getElementById('tally');
console.log('\n  tally text: ' + JSON.stringify(tally.textContent));
check('tally is live', tally.textContent.length > 0);

// --- save produces a real payload ---
let captured = null;
window.URL.createObjectURL = (blob) => { captured = blob; return 'blob:stub'; };
window.confirm = () => true;
doc.querySelector(".sticky button").click();
check('saveDecisions produced a blob', captured !== null);

console.log('\nFAILURES: ' + fails);
process.exit(fails ? 1 : 0);