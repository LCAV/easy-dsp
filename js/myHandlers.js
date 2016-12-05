// Here you can write your own handlers

// function myDataHandler(html, parameters) {
//   // html is the DOM element you can use
//   // Here we just append to this html element the parameters object
//   $(html).append(JSON.stringify(parameters) + '<br />');
//
//   // We must return an object with a method newData
//   return {
//     newData: function (data) {
//       // This code will be executed each time data is sent to this data handler
//       $(html).append(JSON.stringify(data) + '<br />');
//     }
//   }
// }
// dataHandlers.registerNewType('customtype', myDataHandler);
