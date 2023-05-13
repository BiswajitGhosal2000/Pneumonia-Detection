function clearImage() {
    // location.reload();
    window.location = "http://127.0.0.1:5002/";
}
var loadFile = function (event) {
    var image = document.getElementById('uploaded-image');
    image.style.visibility = "visible";
    image.src = URL.createObjectURL(event.target.files[0]);
    document.getElementById('file').style.visibility="hidden";

};