# KOIB RAG — сервер (Docker / systemd)

В этой версии **нет искусственного ограничения 2 ГБ оперативной памяти**.
Фактический расход зависит от модели эмбеддингов, количества векторов,
использования reranker и числа одновременных запросов. Для высоких нагрузок
нужно тестирование на целевой машине.

## Подготовка

```bash
cp .env.server.example .env
# Установите в .env секреты:
#   KOIB_API_KEY=<надёжный случайный ключ>
#   GIGACHAT_CREDENTIALS=<ключ провайдера>
# Если индекс FAISS получен ИЗ ДОВЕРЕННОГО ИСТОЧНИКА,
#   KOIB_TRUST_FAISS_PICKLE=true
```

Создайте индекс на индексаторе и перенесите проверенный архив в `output/`:

```bash
python -m tools.index_transfer unpack --archive index_bundle.zip --output-dir ./output
```

Индекс `.pkl` LangChain FAISS нельзя загружать из неизвестных архивов:
десериализация pickle исполняет произвольный код. Без доверия к индексу
поддерживается только лексический BM25/FTS5.

## Docker Compose

Из корня проекта:

```bash
cd deploy
# читаются ../.env, ../output
# context build = корень проекта; Dockerfile = deploy/Dockerfile
docker compose up --build -d
docker compose ps
```

`BAKE_MODEL=1` загружает Hugging Face E5-модель на этапе сборки; пустой
volume не скрывает закэшированную модель. Для этого нужен сетевой доступ во
время сборки. На сервере можно использовать `BAKE_MODEL=0` в compose.

## Systemd

Установите приложение в `/opt/koib-server`, зависимости в `venv`, настройте
права у пользователя `koib` на рабочие директории и
`sudo cp deploy/koib.service /etc/systemd/system/koib.service`.
После этого `sudo systemctl daemon-reload && sudo systemctl enable --now koib`.
`OMP_NUM_THREADS` можно менять под реальный CPU.

## Проверки

```bash
curl -f http://localhost:8000/health
curl -f http://localhost:8000/ready
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: ВАШ_КЛЮЧ' \
  -d '{"query":"Как проверить сканер?","model_filter":"koib2010"}'
```

`/health` показывает, что процесс работает. `/ready` возвращает 503 без
действительных заполненных баз SQLite. Статус `degraded` означает, что
доступен только FTS-поиск; статус `ready` требует доступных файлов FAISS и
явного разрешения доверенной pickle-десериализации.

Подробные локальные проверки в `QUALITY_REPORT.md`.
