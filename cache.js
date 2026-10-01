import { createKeyv } from '@keyv/redis';

const REDIS_URL = process.env.REDIS_URL || 'redis://127.0.0.1:6379';
const CACHE_NAMESPACE = 'app-cache';
const DEFAULT_CACHE_TTL = 60_000;

const createRedisStore = (namespace) =>
  createKeyv(REDIS_URL, { namespace });

const handleCacheError = (error) => console.error('[Redis Cache Error]', error);

const createLazyCache = () => {
  let store;

  const getStore = () => {
    if (!store) {
      store = createRedisStore(CACHE_NAMESPACE);
      store.on('error', handleCacheError);
    }

    return store;
  };

  return new Proxy({}, {
    get(_target, property) {
      const value = getStore()[property];
      return typeof value === 'function' ? value.bind(getStore()) : value;
    },
  });
};

const cache = createLazyCache();

const createCacheAside =
  (cacheStore, defaultTtl = DEFAULT_CACHE_TTL, onCacheError = handleCacheError) => {
    const inFlightLoads = new Map();

    return async (key, loader, ttl = defaultTtl) => {
      try {
        const cachedValue = await cacheStore.get(key);

        if (cachedValue !== undefined) {
          console.log(`[CACHE HIT] ${key}`);
          return cachedValue;
        }

      } catch (error) {
        onCacheError(error);
      }

      const pendingLoad = inFlightLoads.get(key);
      if (pendingLoad) {
        return pendingLoad;
      }

      console.log(`[CACHE MISS] ${key}`);

      const loadAndCache = (async () => {
        const value = await loader();

        try {
          await cacheStore.set(key, value, ttl);

          console.log(`[CACHE SET] ${key}`);
        } catch (error) {
          onCacheError(error);
        }

        return value;
      })();

      inFlightLoads.set(key, loadAndCache);

      try {
        return await loadAndCache;
      } finally {
        if (inFlightLoads.get(key) === loadAndCache) {
          inFlightLoads.delete(key);
        }
      }
    };
  };

const getOrLoad = createCacheAside(cache);

export default cache;
export { cache, createCacheAside, createRedisStore, DEFAULT_CACHE_TTL, getOrLoad };
