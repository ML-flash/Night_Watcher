import tempfile


def test_database_creation():
    from event_database import EventDatabase
    with tempfile.NamedTemporaryFile() as tmp:
        db = EventDatabase(tmp.name)
        tables = {t['name'] for t in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
        assert 'events' in tables
        assert 'event_observations' in tables
        assert 'match_decisions' in tables


def test_event_crud():
    from event_database import EventDatabase
    with tempfile.NamedTemporaryFile() as tmp:
        db = EventDatabase(tmp.name)
        db.execute(
            "INSERT INTO events (event_id, event_signature, first_seen, last_updated, core_attributes) VALUES (?,?,?,?,?)",
            ['evt1', 'sig', 'now', 'now', '{}']
        )
        result = db.get_event('evt1')
        assert result is not None
        assert result['event_id'] == 'evt1'
