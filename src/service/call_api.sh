curl -i -X POST --http1.1 "localhost:9696/hitec/classify/concepts/bert-classifier/bert/run" \
-H "Content-Type: application/json" \
--data-binary "@src/service/test_request.json"




curl -i -X POST --http1.1 "https://feed-uvl.ifi.uni-heidelberg.de/hitec/classify/concepts/bert-classifier/bert/run" \
-H "Content-Type: application/json" \
--data-binary "@src/service/test_request.json"