import orjson, urllib3, os, asyncio, cProfile, pstats, io
from tqdm import tqdm
from dotenv import load_dotenv
from azure.cosmos.aio import CosmosClient #, PartitionKey
from azure.cosmos import PartitionKey
from openai import OpenAI
load_dotenv()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class CosmosDBUploader:
    def __init__(self, url, key, database_name, container_name):
        self.url = url
        self.key = key
        self.database_name = database_name
        self.container_name = container_name
        self.openai_client = OpenAI()

    async def connect(self):
        self.client = CosmosClient(
            self.url,
            credential=self.key,
            connection_verify=False,  
        )
        self.database = await self.client.create_database_if_not_exists(
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

        self.container = await self.database.create_container_if_not_exists(
            id=self.container_name,
            partition_key=PartitionKey(path="/date_published"),
            offer_throughput=10000,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
        )

    async def _upsert_record(self, record, stats):
        try:
            text = record.get('article_body')
            if text:
                res = self.openai_client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                record['embedding'] = res.data[0].embedding

            await self.container.upsert_item(record)
            stats["ok"] += 1
        except Exception as e:
            stats["fail"] += 1
            print(f"✗ Failed {record.get('id')}: {e}")

    async def upload_file(self, file_path):
        stats = {"ok": 0, "fail": 0}

        with open(file_path, 'rb') as f:
            data = orjson.loads(f.read())

        for record in tqdm(data[:10], desc="Uploading"):
            record['id'] = str(record['id'])
            record['date_published'] = str(record['date_published'])
            await self._upsert_record(record, stats)

        n = stats["ok"] or 1

    def close(self):
        self.openai_client.close()

async def main():
    URL = os.getenv("COSMOS_URL")
    KEY = os.getenv("COSMOS_KEY")
    DATABASE_NAME = "Coresignal_linkedin"
    CONTAINER_NAME = "PostEmbeddings"
    INPUT_FILE = "../data/base.json"

    uploader = CosmosDBUploader(URL, KEY, DATABASE_NAME, CONTAINER_NAME)
    await uploader.connect()
    print(f"Uploading {INPUT_FILE}…")
    await uploader.upload_file(INPUT_FILE)
    uploader.close()

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    asyncio.run(main())
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(20)
    print(s.getvalue())
