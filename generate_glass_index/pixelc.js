// this function writes the coordinates of the cursor when it hovers over the canvas
$(function(){
    $(canvas).mousemove(function(e){
        $('#coordinates').html('x: ' + e.pageX + ' y : ' + e.pageY);
    });
})

// enter the path of the image you want to change
var img = new Image();
img.src = '../sample_images/ml.png';

var canvas = document.getElementById('frame1');
canvas.height = parseInt(img.height);
canvas.width = parseInt(img.width);
var ctx = canvas.getContext('2d');
img.onload = function() {
      ctx.drawImage(img, 0, 0);
      img.style.display = 'none';
};

var cod = []

var flag = 0;
var startdrag = false;
var element = document.getElementById('frame1');

//------------------------------------------------------ the drag movement
element.addEventListener("mousedown", function(){
        flag = 0;
        startdrag = true;
}, false);

var pixel = 4;

element.addEventListener("mousemove", function(e){
        if(startdrag){
          var imd = ctx.getImageData(e.pageX,e.pageY,pixel,pixel);
          var data = imd.data;
          for(var i=0;i<data.length;i++){
              data[i]=255;
          }
          ctx.putImageData(imd, e.pageX, e.pageY);
          for(var i=0;i<pixel;i++){
              cod.push({'x':e.pageX+i,'y':e.pageY+i});
          }
        }
        flag = 1;
}, false);
element.addEventListener("mouseup", function(){
        startdrag = false;
}, false);
//------------------------------------------------------ the drag movement

//one you draw the required shape, download the coordinates of all the points you traversed
var create = document.getElementById('downld');
var textFile = null,
  makeTextFile = function (text) {
          var data = new Blob([text], {type: 'text/plain'});

          // If we are replacing a previously generated file we need to
           // manually revoke the object URL to avoid memory leaks.
           if (textFile !== null) {
                 window.URL.revokeObjectURL(textFile);
         }
         textFile = window.URL.createObjectURL(data);

             // returns a URL you can use as a href
         return textFile;
};

create.addEventListener('click', function () {
        var link = document.createElement('a');
        link.setAttribute('download', '../index/indices.txt');
        var alltext = "";
        for(var item of cod)
            alltext+=item['x']+","+item['y']+"\n";
        link.href = makeTextFile(alltext);
        document.body.appendChild(link);

    // wait for the link to be added to the document
    window.requestAnimationFrame(function () {
    var event = new MouseEvent('click');
    link.dispatchEvent(event);
    document.body.removeChild(link);
    });
}, false);
