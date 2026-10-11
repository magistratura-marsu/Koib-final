# koib-indexer — приложение индексации (мощный компьютер)

1) `cp .env.indexer.example .env` и при необходимости поправить пути/модель.
2) `pip install -r requirements-indexer.txt`  (нужен системный tesseract-ocr +rus).
3) Положить документы в `data/docs/` и собрать индекс:
       python index_app.py ingest --rebuild
4) Проверить и упаковать для сервера:
       python index_app.py verify
       python index_app.py pack --out index_bundle.zip
5) Перенести `index_bundle.zip` на сервер (scp/rsync).
