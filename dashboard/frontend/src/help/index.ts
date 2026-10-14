import type { HelpMap } from "./types"
import { help as overview } from "./overview"
import { help as arms } from "./arms"
import { help as stage1 } from "./stage1"
import { help as units } from "./units"
import { help as m7 } from "./m7"
import { help as glossary } from "./glossary"
import { help as m9 } from "./m9"

export const HELP: HelpMap = { ...glossary, ...overview, ...arms, ...stage1, ...units, ...m7, ...m9 }
