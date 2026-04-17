import orjson, urllib3, os, asyncio, time
from tqdm import tqdm
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

        vector_embedding_policy = {
            "vectorEmbeddings": [
                {
                    "path": "/embedding",
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": 1536
                }
            ]
        }
        indexing_policy = {
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": "/\"_etag\"/?"}],
            "vectorIndexes": [
                {
                    "path": "/embedding",
                    "type": "diskANN"
                }
            ]
        }

        self.container =await self.database.create_container_if_not_exists(
            id=self.container_name,
            partition_key=PartitionKey(path="/date_published"),
            offer_throughput=10000, # max request unit persec
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
        )
    
    async def _upsert_with_semaphore(self, record, sem, stats):
        async with sem:
            try:
                text = record.get('article_body')
                if text:
                    t0 = time.perf_counter()
                    res = await self.openai_client.embeddings.create(
                        input=text,
                        model="text-embedding-3-small"
                    )
                    stats["embed_time"] += time.perf_counter() - t0
                    record['embedding'] = res.data[0].embedding

                t0 = time.perf_counter()
                await self.container.upsert_item(record)
                stats["upsert_time"] += time.perf_counter() - t0
                stats["ok"] += 1
            except Exception as e:
                stats["fail"] += 1
                print(f"✗ Failed {record.get('id')}: {e.status_code} {e.message}")

    async def upload_file(self, file_path):
        sem = asyncio.Semaphore(self.CONCURRENCY)
        stats = {"ok": 0, "fail": 0, "embed_time": 0.0, "upsert_time": 0.0}
        tasks = []
        t_total = time.perf_counter()

        with open(file_path, 'rb') as f:
            data = orjson.loads(f.read())
            for record in tqdm(data, desc="Uploading"):
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

        total = time.perf_counter() - t_total
        n = stats["ok"] or 1
        print(f"\n--- Profile ---")
        print(f"Total time   : {total:.1f}s")
        print(f"Embed time   : {stats['embed_time']:.1f}s  ({stats['embed_time']/total*100:.1f}%)")
        print(f"Upsert time  : {stats['upsert_time']:.1f}s  ({stats['upsert_time']/total*100:.1f}%)")
        print(f"Records ok   : {stats['ok']:,}  |  failed: {stats['fail']:,}")
        print(f"Avg embed/rec: {stats['embed_time']/n*1000:.1f}ms")
        print(f"Avg upsert/rec: {stats['upsert_time']/n*1000:.1f}ms")
        print(f"Throughput   : {n/total:.1f} records/s")

    async def close(self):
        await self.client.close()
        await self.openai_client.close()

async def main():

    URL = os.getenv("COSMOS_URL")
    KEY = os.getenv("COSMOS_KEY")
    DATABASE_NAME = "Coresignal_linkedin"
    CONTAINER_NAME = "PostEmbeddings"
    INPUT_FILE = "../data/base.json"

    uploader = CosmosDBUploader(URL, KEY, DATABASE_NAME, CONTAINER_NAME)
    await uploader.connect()
    print(f"Uploading {INPUT_FILE} with concurrency={uploader.CONCURRENCY}…")
    await uploader.upload_file(INPUT_FILE)
    await uploader.close()

if __name__ == "__main__":
    asyncio.run(main())