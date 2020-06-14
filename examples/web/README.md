<div align="center">
  <h1><code>WASM + Rustlight</code></h1>
</div>

An example of compiling [Rustlight](https://github.com/beltegeuse/rustlight) with WASM and able to run it inside a web-browser.

## Usage

You can compile the rust-glue of Rustlight with
```
wasm-pack build --target=web
```
Note that the following command generate a WASM file inside `pkg` directory. 

Then, due to browser restriction, you need to start a websever to access to `index.html`. The easiest way is to run this command:
```
python3 -m http.server
```

You can then edit the PBRT file in the textbox. In this version, only geometric/material/light information has an impact on the final image. After clicking the `Render!` button, the image will render with path tracing (NEE and Russian roulette) with one core.  

![screenshot](https://raw.githubusercontent.com/beltegeuse/rustlight-web/master/assets/screenshot.png)

You can stop the rendering with the `Stop!` button. 

## TODO

- Export wasm to a webpage (AWS)
- [Add multi-core support](https://github.com/rustwasm/wasm-bindgen/issues/2175)
- Improve interface and JS code
- Extend scene support