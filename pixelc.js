$(function(){
    $(canvas).mousemove(function(e){
        $('#coordinates').html('x: ' + e.pageX + ' y : ' + e.pageY);
    });
})

var img = new Image();
img.src = 'sample_images/ml.png';
var canvas = document.getElementById('frame1');
canvas.height = parseInt(img.height);
canvas.width = parseInt(img.width);
var ctx = canvas.getContext('2d');
img.onload = function() {
      ctx.drawImage(img, 0, 0);
      img.style.display = 'none';
};

var cod = []
//var id = ctx.createImageData(1,1); // only do this once per page
//var d  = id.data;                        // only do this once per page

var flag = 0;
var startdrag = false;
var element = document.getElementById('frame1');
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
    if(flag === 1 && startdrag){
       //console.log(cod);
    }
        startdrag = false;
}, false);

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
        link.setAttribute('download', 'info.txt');
        var alltext = "";
        for(var item of cod)
            //console.log(item);
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
