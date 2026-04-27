## Pixel Pitch

Link: https://hletrd.github.io/pixelpitch/

Original work by Maik Riechert: https://letmaik.github.io/pixelpitch/
 ([repo](https://github.com/letmaik/pixelpitch))

Lists current cameras with their physical pixel sizes (pixel pitch). As Maik's [original work](https://github.com/letmaik/pixelpitch) seems to be no longer maintained, I've forked the project and updated the data.

### Data Sources

Camera data is aggregated from multiple sources:

- **[geizhals.eu](https://geizhals.eu)** — Primary source for DSLR, mirrorless, rangefinder, fixed-lens, camcorder, and action camera data (scraped via browser automation)
- **[Imaging Resource](https://www.imaging-resource.com)** — Per-camera spec pages with explicit pixel pitch measurements
- **[Apotelyt](https://apotelyt.com)** — Per-camera sensor-pixel pages (good gap-filler for recent models)
- **[GSMArena](https://www.gsmarena.com)** — Smartphone main-camera sensor specs
- **[CineD](https://www.cined.com)** — Cinema camera database
- **[openMVG/CameraSensorSizeDatabase](https://github.com/openMVG/CameraSensorSizeDatabase)** — MIT-licensed CSV with bulk sensor data (also used as proxy for digicamdb.com)

Note that pixel pitch is calculated from resolution and sensor size, with a look up table if exact sensor size is not given (but instead a common size name). In 2015 geizhals.at added pixel pitch to their website as well, it may happen that the pixel pitch values slightly differ due to different formulas used.

### Alternative Source Fetching

Individual sources can be fetched independently:

```
python pixelpitch.py source <name> [--limit N] [--out DIR]
```

Available sources: `openmvg`, `digicamdb`, `imaging-resource`, `apotelyt`, `gsmarena`, `cined`
