import { setupServer } from "msw/node";
import { handlers } from "./handlers";

/**
 * MSW server for Node (Vitest). Default handlers cover the happy
 * path of every endpoint the components touch; individual tests
 * can ``server.use(...)`` to override per-test.
 */
export const server = setupServer(...handlers);
