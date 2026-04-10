import json, urllib3, os
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, PartitionKey, exceptions
load_dotenv()

# solve warning issue
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class CosmosDBUploader:
    def __init__(self, url, key, database_name, container_name):
        self.url = url
        self.key = key
        self.database_name = database_name
        self.container_name = container_name

    def connect(self):
        self.client = CosmosClient(
            self.url,
            credential=self.key,
            connection_verify=False,
        )
        self.database = self.client.create_database_if_not_exists(
            id=self.database_name
        )

        self.container = self.database.create_container_if_not_exists(
            id=self.container_name,
            partition_key=PartitionKey(path="/id"),
            offer_throughput=10000 # request unit persec
        )

    def upload_file(self, file_path):

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record['id'] = str(record['id'])

                try:
                    self.container.upsert_item(record)
                except exceptions.CosmosHttpResponseError as e:
                    print(f"Failed to insert {record['id']}: {e}")
                    
        print("Uploaded")

if __name__ == "__main__":

    URL = os.getenv("COSMOS_URL")
    KEY = os.getenv("COSMOS_KEY")
    DATABASE_NAME = "Coresignal_linkedin"
    CONTAINER_NAME = "CleanedPosts"
    INPUT_FILE = "./cleaned_output.json"

    uploader = CosmosDBUploader(URL, KEY, DATABASE_NAME, CONTAINER_NAME)
    uploader.connect()
    uploader.upload_file(INPUT_FILE)
