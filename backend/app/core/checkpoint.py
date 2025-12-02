import json
from typing import Any, Dict, Iterator, List, Optional, Tuple
from datetime import datetime
from pymongo import MongoClient
from app.config import settings


class MongoDBCheckpointer:
    """MongoDB-based checkpointer for LangGraph workflows (synchronous)"""
    
    def __init__(self):
        self.collection_name = "langgraph_checkpoints"
        self._client: Optional[MongoClient] = None
        self._db = None
    
    def _get_db(self):
        """Get database instance (synchronous pymongo)"""
        if self._client is None:
            self._client = MongoClient(settings.MONGO_URL)
            self._db = self._client[settings.MONGO_DB_NAME]
        return self._db
    
    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        new_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save a checkpoint to MongoDB"""
        db = self._get_db()
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        
        checkpoint_doc = {
            "thread_id": thread_id,
            "checkpoint": checkpoint,
            "metadata": metadata or {},
            "new_versions": new_versions or {},
            "created_at": datetime.utcnow(),
        }
        
        # Upsert: replace existing checkpoint for this thread_id
        db[self.collection_name].update_one(
            {"thread_id": thread_id},
            {"$set": checkpoint_doc},
            upsert=True
        )
        
        return checkpoint_doc
    
    def get(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve a checkpoint from MongoDB"""
        db = self._get_db()
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        
        doc = db[self.collection_name].find_one({"thread_id": thread_id})
        
        if doc:
            return {
                "checkpoint": doc.get("checkpoint", {}),
                "metadata": doc.get("metadata", {}),
            }
        return None
    
    def list(
        self,
        config: Optional[Dict[str, Any]] = None,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """List checkpoints from MongoDB"""
        db = self._get_db()
        
        query = {}
        if config and "configurable" in config:
            thread_id = config["configurable"].get("thread_id")
            if thread_id:
                query["thread_id"] = thread_id
        
        if filter:
            query.update(filter)
        
        cursor = db[self.collection_name].find(query).sort("created_at", -1)
        
        if limit:
            cursor = cursor.limit(limit)
        
        for doc in cursor:
            yield {
                "checkpoint": doc.get("checkpoint", {}),
                "metadata": doc.get("metadata", {}),
                "thread_id": doc.get("thread_id"),
            }

