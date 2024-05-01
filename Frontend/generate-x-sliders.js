import data from './default-landmarks.json' assert { type: 'json' };

for (let i=0; i<68; i++) {
    let landmark_id = `landmark-${i}`;
    let default_x = data[landmark_id]?.x || 0.5;
    let html = `<div class="slidecontainer"><input type="range" min="0" max="1" step="0.01" value="${default_x}" class="slider" id="slider-l${i}-x"><p style="display:inline;">Landmark ${i} X Value:</p><p id="l${i}-x" style="display:inline;"></p></div>`;
    $('#sliders-x').append(html);
}
