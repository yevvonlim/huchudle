import data from './default-landmarks.json' assert { type: 'json' };

for (let i=0; i<68; i++) {
    let landmark_id = `landmark-${i}`;
    let default_y = data[landmark_id]?.y || 0.5;
    let html = `<div class="slidecontainer"><input type="range" min="0" max="1" step="0.01" value="${default_y}" class="slider" id="slider-l${i}-y"><p style="display:inline;">Landmark ${i} Y Value:</p><p id="l${i}-y" style="display:inline;"></p></div>`;
    $('#sliders-y').append(html);
}
