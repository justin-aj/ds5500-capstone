# DS 5500 Capstone

Detecting item transitions across a boundary from RFID reads.

Author: Ajin Frank Justin

## Abstract

Inventory systems use RFID readers to record an item as received or shipped when its tag crosses a boundary like a doorway. The reader never sees the crossing, though. It only reports reads, and tags that turn back or sit parked nearby get read too. I'm building a classifier that decides, for each pass of a tag, whether it crossed, using reads from eight antennas in three groups around one boundary. My 19 features compare the antenna groups with each other (which groups heard the tag, their share of reads, the order and the signal gaps), so they don't depend on item speed. My logistic regression baseline, trained only on the first data collection, found 998 of 999 straight passes in a second, separate recording, with 16 false alarms out of 41,327 parked tags, and 24 false alarms out of 3,897 parked tags in a third. Turn-backs and outward crossings are still untested on new recordings.
