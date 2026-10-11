from api.routes.query import QueryRequest

def test_validation_alias():
    assert QueryRequest(query="test", validate=False).validate_answer is False
    assert QueryRequest(query="test").validate_answer is True
