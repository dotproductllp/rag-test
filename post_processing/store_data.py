import json, urllib3, os, asyncio
from dotenv import load_dotenv
from azure.cosmos.aio import CosmosClient
from azure.cosmos import PartitionKey
from openai import AsyncOpenAI
load_dotenv()

# solve warning issue
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class CosmosDBUploader:
    def __init__(self, url, key, database_name, container_name):
        self.url = url
        self.key = key
        self.database_name = database_name
        self.container_name = container_name

        # tuning params
        self.CONCURRENCY = 32
        self.CHUNK_SIZE = 10000

        self.openai_client = AsyncOpenAI()

    async def connect(self):
        self.client = CosmosClient(
            self.url,
            credential=self.key,
            connection_verify=False,
            enable_bulk=True,
            retry_total=30, 
            retry_backoff_max=120
        )
        self.database =await self.client.create_database_if_not_exists(
            id=self.database_name
        )

        self.container =await self.database.create_container_if_not_exists(
            id=self.container_name,
            partition_key=PartitionKey(path="/date_published"),
            offer_throughput=10000, # request unit persec
        )
    
    async def _upsert_with_semaphore(self, record, sem, stats):
        async with sem:
            try:
                text = record.get('article_body')
                if text:
                    res = await self.openai_client.embeddings.create(
                        input=text,
                        model="text-embedding-3-small"
                    )
                    record['embedding'] = res.data[0].embedding
                await self.container.upsert_item(record)
                stats["ok"] += 1
            except Exception as e:
                stats["fail"] += 1
                print(f"✗ Failed {record.get('id')}: {e.status_code} {e.message}")

    async def upload_file(self, file_path):
        sem = asyncio.Semaphore(self.CONCURRENCY)
        stats = {"ok": 0, "fail": 0}
        tasks = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record['id'] = str(record['id'])
                record['date_published'] = str(record['date_published'])

                tasks.append(
                    asyncio.create_task(
                        self._upsert_with_semaphore(record, sem, stats)
                    )
                )

                # chunk flush
                if len(tasks) >= self.CHUNK_SIZE:
                    await asyncio.gather(*tasks)
                    tasks.clear()
                    print(f"↑ Uploaded {stats['ok']:,} records so far…")

        if tasks:
            await asyncio.gather(*tasks)
                    
        print("Uploaded")

    async def close(self):
        await self.client.close()
        await self.openai_client.close()

async def main():

    URL = os.getenv("COSMOS_URL")
    KEY = os.getenv("COSMOS_KEY")
    DATABASE_NAME = "Coresignal_linkedin"
    CONTAINER_NAME = "CleanedPostsWithEmbeddingsTest"
    INPUT_FILE = "./cleaned_output.json"

    uploader = CosmosDBUploader(URL, KEY, DATABASE_NAME, CONTAINER_NAME)
    await uploader.connect()
    print(f"Uploading {INPUT_FILE} with concurrency={uploader.CONCURRENCY}…")
    await uploader.upload_file(INPUT_FILE)
    await uploader.close()

if __name__ == "__main__":
    asyncio.run(main())