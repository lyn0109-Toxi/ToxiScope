from pathlib import Path
from streamlit.testing.v1 import AppTest
APP = str(Path(__file__).resolve().parents[1] / 'streamlit_app.py')

def test_ntr_navigation_evaluate_reset_and_return():
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    assert app.checkbox(key='external_consent:').value is False
    app.radio(key='workspace').set_value('Toxicity Evidence Review').run()
    assert not app.exception
    next(b for b in app.button if b.label == 'Review evidence gaps').click().run()
    assert not app.exception
    assert any(m.value == 'R0' for m in app.metric)
    next(b for b in app.button if b.label == 'Clear NTR workspace').click().run()
    assert not app.exception
    app.radio(key='workspace').set_value('Strategy Dashboard').run()
    assert any(b.label == 'Calculate f2 and simulation' for b in app.button)
    app.radio(key='workspace').set_value('Toxicity Evidence Review').run()
    assert not app.exception
    assert len(app.session_state['ntr_docs']) == 0

def test_invalid_manual_input_and_query_consent_reset():
    app = AppTest.from_file(APP).run(timeout=30)
    app.checkbox(key='external_consent:').check().run()
    next(t for t in app.text_input if t.label == 'Compound Name').set_value('PRIVATE-NEW').run()
    assert not app.checkbox(key='external_consent:PRIVATE-NEW').value
    app.radio(key='workspace').set_value('Toxicity Evidence Review').run()
    next(t for t in app.text_area if t.label == 'Reviewed NTR input (JSON)').set_value('{bad')
    next(b for b in app.button if b.label == 'Review evidence gaps').click().run()
    assert not app.exception
    assert any('Evaluation blocked' in e.value for e in app.error)

def test_existing_qsar_be_and_mismatched_context():
    app = AppTest.from_file(APP).run(timeout=30)
    next(t for t in app.text_input if t.label == 'Compound Name').set_value('Synthetic public aspirin')
    app.text_area(key='smiles').set_value('CC(=O)OC1=CC=CC=C1C(=O)O')
    next(b for b in app.button if b.label == '🚀 Run Regulatory Assessment').click().run()
    assert not app.exception
    assert app.session_state['evidence_package']['assessment']
    next(b for b in app.button if b.label == 'Calculate f2 and simulation').click().run()
    assert not app.exception
    assert app.session_state['be_result'].f2 > 50
    app.radio(key='workspace').set_value('Toxicity Evidence Review').run()
    app.text_input(key='ntr_compound_name').set_value('Different project').run()
    assert any('Structural context is excluded' in w.value for w in app.warning)
    assert not app.exception
