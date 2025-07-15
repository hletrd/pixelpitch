## Pixel Pitch

Link: https://hletrd.github.io/pixelpitch/

Original work by Maik Riechert: https://letmaik.github.io/pixelpitch/
 ([repo](https://github.com/letmaik/pixelpitch))

Lists current cameras with their physical pixel sizes (pixel pitch). As Maik's [original work](https://github.com/letmaik/pixelpitch) seems to be no longer maintained, I've forked the project and updated the data.

Camera data is read from http://geizhals.at. Note that pixel pitch is calculated from resolution and sensor size, with a look up table if exact sensor size is not given (but instead a common size name). In 2015 geizhals.at added pixel pitch to their website as well, it may happen that the pixel pitch values slightly differ due to different formulas used.
