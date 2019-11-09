$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    //$('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                //$('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            };
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#btn-preprocessing').show();
        $('#result').text('Step 1: Preprocess your input.');
        readURL(this);
    });

    // preprocess
    var bbox;
    var img_path;
    function set_bbox(value, img){
        bbox = value;
        img_path = img;
    }

    $("#btn-preprocessing" ).click(function() {
        var form_data = new FormData($('#upload-file')[0]);
        console.log("Get form data");
        // Show loading animation
        $(this).hide();
        $('.loader').show();
        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/bbox_predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                set_bbox(data['bbox'], data.filename);
                $('#imagePreview').html(
                    "<img src='/static/" + data.filename +  "' style='max-width: 100%;'/>"
                );
                $('#result').fadeIn(600);
                $('#result').text('Step 2: Make a prediction based on pre-processed image.');
            }
        });

    });

    // Predict
    var predict_result;
    function set_predict_result(value){
        predict_result = value;
    }

    $('#btn-predict').click(function () {
        if(bbox == null) {
            alert('Not preprocess yet !!!!!!!');
            return;
        }

        var form_data = new FormData($('#upload-file')[0]);
        form_data.append('bbox', bbox);
        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                console.log(data);
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text('Result:  Left KLG=' + data.left + '; Right KLG=' + data.right);
                set_predict_result(data);
            }
        });
    });

});
