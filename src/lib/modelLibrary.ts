import type { SavedModel } from "../types";

const DATABASE_NAME = "watchneuron-model-library";
const DATABASE_VERSION = 1;
const MODEL_STORE = "models";

function openDatabase() {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(MODEL_STORE)) {
        const store = database.createObjectStore(MODEL_STORE, { keyPath: "id" });
        store.createIndex("createdAt", "createdAt");
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("无法打开模型库"));
  });
}

async function runRequest<T>(
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest<T>,
) {
  const database = await openDatabase();
  return new Promise<T>((resolve, reject) => {
    const transaction = database.transaction(MODEL_STORE, mode);
    const request = operation(transaction.objectStore(MODEL_STORE));
    let result: T;
    request.onsuccess = () => {
      result = request.result;
    };
    request.onerror = () => reject(request.error ?? new Error("模型库操作失败"));
    transaction.oncomplete = () => {
      database.close();
      resolve(result);
    };
    transaction.onabort = () => {
      database.close();
      reject(transaction.error ?? new Error("模型库事务失败"));
    };
  });
}

export async function listSavedModels() {
  const models = await runRequest<SavedModel[]>("readonly", (store) => store.getAll());
  return models.sort((left, right) => right.createdAt - left.createdAt);
}

export function saveModelRecord(model: SavedModel) {
  return runRequest<IDBValidKey>("readwrite", (store) => store.put(model));
}

export function deleteModelRecord(id: string) {
  return deleteModelRecords([id]);
}

export async function deleteModelRecords(ids: string[]) {
  const uniqueIds = [...new Set(ids)];
  if (!uniqueIds.length) return;

  const database = await openDatabase();
  return new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(MODEL_STORE, "readwrite");
    const store = transaction.objectStore(MODEL_STORE);
    uniqueIds.forEach((id) => store.delete(id));
    transaction.oncomplete = () => {
      database.close();
      resolve();
    };
    transaction.onabort = () => {
      database.close();
      reject(transaction.error ?? new Error("无法删除模型"));
    };
  });
}

export async function renameModelRecord(id: string, name: string) {
  const database = await openDatabase();
  return new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(MODEL_STORE, "readwrite");
    const store = transaction.objectStore(MODEL_STORE);
    const request = store.get(id);
    request.onsuccess = () => {
      const model = request.result as SavedModel | undefined;
      if (!model) {
        transaction.abort();
        reject(new Error("模型不存在"));
        return;
      }
      store.put({ ...model, name });
    };
    request.onerror = () => reject(request.error ?? new Error("无法读取模型"));
    transaction.oncomplete = () => {
      database.close();
      resolve();
    };
    transaction.onabort = () => {
      database.close();
      reject(transaction.error ?? new Error("无法重命名模型"));
    };
  });
}
