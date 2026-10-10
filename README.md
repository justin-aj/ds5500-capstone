# DS 5500 Capstone

Detecting item transitions from RFID read streams.

Author: Ajin Frank Justin

## Summary

RFID readers can identify tagged items at a distance, which makes them useful for tracking inventory in logistics, retail, and manufacturing. One key event in these settings is a transition, when an item crosses a boundary like a dock door or portal, which might trigger a system to record that the item entered or left a zone. The reader doesn't actually see the crossing happen, though. What it reports is a stream of reads, each containing a tag ID, the antenna that picked it up, the signal strength (RSSI), RF phase, and a timestamp. The tricky part is that items can approach the boundary and turn back, or just sit near it, or get read from far away without ever crossing. So one strong read doesn't tell you much.

The goal here is to classify each tag pass as an inward crossing (`in`), outward crossing (`out`), or no crossing (`none`) based on reads from antennas positioned around a boundary. I'm working with two RFID datasets and plan to compare tree-based models, neural networks, and sequence models. I'll also test how well a model trained on one dataset works when applied to the other.
