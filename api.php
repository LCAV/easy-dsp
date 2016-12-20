<?php
header('Content-type: application/json');

if (isset($_GET['restart'])) {
  exec('./stop.sh');
  exec('./start.sh logs.txt');
  exit('{"success": true}');
}
if (isset($_GET['stop'])) {
  exec('./stop.sh');
  exit('{"success": true}');
}
?>
