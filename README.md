## AI template
A structural framework for coding and vizualizing AIs, for my videos.

### Setup
Simply add this crate to your project's Cargo.toml.
```toml
[dependencies]
ai_template = "*"
```

If displaying the simulation vizualization, you will need a device with Vulkan supported.  
If encoding, you will need something a video encoder like FFmpeg to concatenate generated frames into a video.

Note the generated frames are not limited by your screen size.