function getData() {
    var KodeRegistrasi = '<?= $data['Value'] ?>';
    var html = '';

    $.ajax({
        url: '<?= Yii::$app->urlManager->createUrl(["/setting/umum/registrasi/detail"]) ?>',
        type: 'GET',
        data: {
            noReg: KodeRegistrasi
        },
        dataType: 'JSON',
        success: function(data) {
            if (data === "Tidak ada Nomor Registrasi") {
                swal({
                    title: "Maaf!",
                    text: "Kode Registrasi Tidak Ditemukan!",
                    icon: "error", // Use "error" instead of "success" for a negative message.
                    button: "Daftar Ulang",
                }).then(function() {
                    registrasi();
                });
            } else {
                // Handle successful data retrieval
                var $provinceId = data['provinsi'];

                // Fetch province name based on the province ID
                $.ajax({
                    url: '<?= Yii::$app->urlManager->createUrl(["/setting/umum/registrasi/GetProv"]) ?>',
                    type: 'GET',
                    data: { provinceId: $provinceId },
                    dataType: 'JSON',
                    success: function(provinceData) {
                        var provinceName = provinceData['NamaPropinsi'] || 'N/A';
                        // Update the UI with retrieved data
                        $('#perpusName').html(data['namaPerpus']);
                        $('#perpusJenis').html(data['jenisPerpus']);
                        $('#perpusNegara').html(data['negara']);
                        $('#perpusProv').html(provinceName);
                        $('#perpusKode').html(data['kodeRegis']);
                    },
                    error: function() {
                        console.error('Failed to fetch province data.');
                        $('#perpusProv').html('Error fetching province');
                    }
                });
            }
        },
        error: function(xhr, status, error) {
            console.error('Failed to fetch registration details:', error);
            swal({
                title: "Error!",
                text: "Failed to retrieve data from the server.",
                icon: "error",
                button: "Try Again",
            });
        }
    });
}
