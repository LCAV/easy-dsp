// The dropdown menu
var dropdownContainer = $('#div-examples-dropdown');

var examplesFolder = '../examples/';

// Load the list of examples
$.getJSON(examplesFolder + '/list.json', function (examples) {
  _.forEach(examples, function (example) {
    var e = $('<a class="dropdown-item" href="#">' + example.name + '</a>');
    e.click(function (e) {
      e.preventDefault();
      console.info("Load", example.file);
      $.get(examplesFolder + '/' + example.file, function (file) {
        aceEditor.setValue(file, -1);
      });
    });
    dropdownContainer.append(e);
  });
});
