# koib-server — серверная сборка

Только обслуживание запросов. Индексации нет.

1) `cp .env.server.example .env` и заполнить секреты (GigaChat и VK).
2) Установить зависимости:
       pip install -r requirements-torch-cpu.txt
       pip install -r requirements-server.txt
3) Положить индекс: перенесите `index_bundle.zip` с индексатора и распакуйте:
       python -m tools.index_transfer unpack --archive index_bundle.zip --output-dir ./output
4) Запуск:
       python serve_app.py --serve
   Проверка готовности: `GET /ready`.
