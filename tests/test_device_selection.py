import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import time
import unittest
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialogButtonBox
from mes_vision.operation.camera import D405Camera
from mes_vision.operation.catalog import default_equipment
from mes_vision.operation.setup_dialogs import EquipmentDialog
from mes_vision.operation.device_discovery import collect_devices, discover_devices

APP=QApplication.instance() or QApplication([])


def mode(w=640,h=480,fps=30): return {'width':w,'height':h,'fps':fps}
def camera(serial='camera-A',profiles=None):
    return {'serial':serial,'name':'Intel RealSense D405','profiles':[mode()] if profiles is None else profiles,'profile_error':None}
def discovery(cameras=None,ports=None,errors=None):
    return {'cameras':cameras or [],'ports':ports or [],'errors':errors or {}}


def hung_collector(output):
    output.put(('ports',[{'port':'COM7','description':'USB Serial'}],None))
    time.sleep(30)


def failed_collector(output):
    output.put(('ports',[],None)); output.put(('cameras',[],'driver missing'))


class EnumerationTests(unittest.TestCase):
    def sdk(self, profiles):
        sensor=NS(get_stream_profiles=lambda:profiles)
        device=NS(get_info=lambda key:{'name':'D405','serial':'A'}[key],query_sensors=lambda:[sensor])
        other=NS(get_info=lambda key:'D435')
        return NS(camera_info=NS(name='name',serial_number='serial'),stream=NS(color='color'),format=NS(rgb8='rgb8'),context=lambda:NS(query_devices=lambda:[other,device]))

    def profile(self,w=640,h=480,fps=30,stream='color',format='rgb8'):
        return NS(stream_type=lambda:stream,format=lambda:format,
            as_video_stream_profile=lambda:NS(width=lambda:w,height=lambda:h,fps=lambda:fps))

    def test_enumeration_filters_color_rgb8_and_deduplicates_without_starting_stream(self):
        sdk=self.sdk([self.profile(),self.profile(),self.profile(fps=15),self.profile(stream='depth'),self.profile(format='bgr8')])
        rows=D405Camera.devices(sdk=sdk)
        self.assertEqual(len(rows),1); self.assertEqual(rows[0]['serial'],'A')
        self.assertEqual(rows[0]['profiles'],[mode(fps=30),mode(fps=15)])

    def test_known_camera_is_preserved_when_profile_read_fails(self):
        sdk=self.sdk([NS(stream_type=Mock(side_effect=RuntimeError('unplugged')))])
        row=D405Camera.devices(sdk=sdk)[0]
        self.assertEqual(row['serial'],'A'); self.assertIsNone(row['profiles']); self.assertEqual(row['profile_error'],'unplugged')

    def test_empty_profiles_distinguished_from_failed_query(self):
        self.assertEqual(D405Camera.devices(sdk=self.sdk([]))[0]['profiles'],[])

    def test_camera_failure_keeps_ports_and_does_not_connect(self):
        output=Mock()
        with patch('mes_vision.robot.magician.Magician.ports',return_value=[{'port':'COM7','description':'USB'}]),patch.object(D405Camera,'devices',side_effect=RuntimeError('driver missing')):
            collect_devices(output,driver="d405")
        self.assertEqual(output.put.call_args_list[0].args[0],('ports',[{'port':'COM7','description':'USB'}],None))
        self.assertEqual(output.put.call_args_list[1].args[0],('cameras',[],'driver missing'))

    def test_stalled_driver_is_bounded_and_partial_results_survive(self):
        before=time.monotonic(); result=discover_devices(timeout=2.,collector=hung_collector)
        self.assertLess(time.monotonic()-before,5.); self.assertEqual(result['ports'][0]['port'],'COM7')
        self.assertIn('cameras',result['errors']); self.assertNotIn('ports',result['errors'])

    def test_explicit_driver_error_is_not_reported_as_no_devices(self):
        result=discover_devices(timeout=5.,collector=failed_collector)
        self.assertEqual(result['errors'],{'cameras':'driver missing'})


class DeviceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.equipment=default_equipment(); self.equipment['camera'].update(driver='d405',width=640,height=480); self.equipment['camera']['serial']='camera-A'; self.equipment['robot']['port']='COM7'
        self.original=deepcopy(self.equipment); self.dialog=EquipmentDialog(self.equipment)

    def tearDown(self):
        deadline=time.monotonic()+3
        while self.dialog.discovery_read.busy and time.monotonic()<deadline:
            APP.processEvents(); time.sleep(.005)
        self.dialog.close(); self.dialog.deleteLater(); APP.processEvents()

    def test_offline_saved_identifiers_and_legacy_modes_survive(self):
        d=self.dialog; d.devices_found(discovery()); d.save()
        self.assertEqual(d.result(),1,d.message.text())
        self.assertEqual(d.value['camera'],self.original['camera']); self.assertEqual(d.value['robot']['port'],'COM7')
        self.assertEqual(self.equipment,self.original)
        self.assertIn('확인되지 않음',d.serial.currentText())

    def test_discovery_does_not_auto_select_a_different_device(self):
        d=self.dialog; d.devices_found(discovery([camera('B')],[{'port':'COM8','description':'USB Serial'}]))
        self.assertEqual(d.serial.identifier(),'camera-A'); self.assertEqual(d.port.identifier(),'COM7')
        d.serial.setCurrentIndex(0); d.devices_found(discovery([camera('B')]))
        self.assertEqual(d.serial.identifier(),'')

    def test_selected_device_modes_replace_presets_and_saved_identifiers_use_item_data(self):
        d=self.dialog; d.devices_found(discovery([camera()],[{'port':'COM7','description':'USB Serial Device'}]))
        self.assertEqual(d.resolution.count(),1); self.assertEqual(d.fps.count(),1)
        d.save(); self.assertEqual(d.result(),1,d.message.text())
        self.assertEqual(d.value['camera']['serial'],'camera-A'); self.assertEqual(d.value['robot']['port'],'COM7')

    def test_unsupported_existing_mode_is_preserved_but_cannot_be_saved(self):
        d=self.dialog; d.devices_found(discovery([camera(profiles=[mode(1280,720,15)])]))
        self.assertEqual(d.selected_resolution(),(640,480)); self.assertEqual(d.fps.currentData(),30)
        d.save(); self.assertEqual(d.result(),0); self.assertIn('지원되지 않습니다',d.message.text())
        d.resolution.setCurrentIndex(d.resolution.findData('1280x720'))
        self.assertEqual(d.fps.currentData(),15); d.save(); self.assertEqual(d.result(),1,d.message.text())
        self.assertNotEqual(d.value['camera']['acquisition_revision'],self.original['camera']['acquisition_revision'])

    def test_switching_device_preserves_selection_and_checks_new_rates(self):
        d=self.dialog; d.devices_found(discovery([camera(),camera('B',[mode(fps=15)])]))
        d.serial.setCurrentIndex(d.serial.findData('B'))
        self.assertEqual(d.fps.currentData(),30); self.assertFalse(d.selected_mode_supported())
        d.fps.setCurrentIndex(d.fps.findData(15)); self.assertTrue(d.selected_mode_supported())

    def test_refresh_preserves_unsaved_mode_and_port_selection(self):
        d=self.dialog; result=discovery([camera(profiles=[mode(),mode(1280,720,15)])],[{'port':'COM8','description':'USB'}])
        d.devices_found(result); d.resolution.setCurrentIndex(d.resolution.findData('1280x720'))
        d.port.setCurrentIndex(d.port.findData('COM8')); d.devices_found(result)
        self.assertEqual(d.selected_resolution(),(1280,720)); self.assertEqual(d.fps.currentData(),15); self.assertEqual(d.port.identifier(),'COM8')

    def test_failed_rescan_invalidates_previous_profiles(self):
        d=self.dialog; d.devices_found(discovery([camera()]))
        d.discovery_failed('driver disappeared')
        self.assertNotIn('camera-A',d.discovered_cameras); self.assertGreater(d.resolution.count(),1)
        self.assertEqual(d.serial.identifier(),'camera-A'); self.assertIn('검색 실패',d.search_status.text())

    def test_successful_query_with_no_rgb8_modes_blocks_save(self):
        d=self.dialog; d.devices_found(discovery([camera(profiles=[])])); d.save()
        self.assertEqual(d.result(),0); self.assertIn('RGB8',d.mode_hint.text())

    def test_profile_query_failure_allows_offline_setup_with_explicit_unverified_hint(self):
        d=self.dialog; entry=camera(); entry.update(profiles=None,profile_error='disconnected')
        d.devices_found(discovery([entry])); d.save()
        self.assertEqual(d.result(),1); self.assertIn('확인하지 못했습니다',d.mode_hint.text()); self.assertEqual(d.mode_hint.toolTip(),'disconnected')

    def test_manual_offline_identifier_is_not_confused_with_display_label(self):
        d=self.dialog; d.serial.setEditText('manual-serial'); d.port.setEditText('COM99'); d.save()
        self.assertEqual(d.value['camera']['serial'],'manual-serial'); self.assertEqual(d.value['robot']['port'],'COM99')

    def test_rescan_error_status_does_not_hide_successful_ports(self):
        d=self.dialog; d.devices_found(discovery(ports=[{'port':'COM8','description':'USB'}],errors={'cameras':'missing SDK'}))
        self.assertGreater(d.port.findData('COM8'),0); self.assertIn('검색 실패',d.search_status.text()); self.assertIn('missing SDK',d.search_status.toolTip())

    def test_async_search_coalesces_blocks_save_and_closes_without_destroying_thread(self):
        d=self.dialog; d.show(); beats=[]; timer=QTimer(); timer.timeout.connect(lambda:beats.append(1)); timer.start(10)
        def slow(**kwargs): time.sleep(.15); return discovery([camera()])
        with patch('mes_vision.operation.device_controls.discover_devices',side_effect=slow) as read:
            d.search_devices(); d.search_devices(); self.assertTrue(d.discovery_read.busy)
            self.assertFalse(d.save_box.button(QDialogButtonBox.Save).isEnabled())
            d.save(); self.assertEqual(d.result(),0)
            d.reject(); self.assertTrue(d.isVisible())
            deadline=time.monotonic()+2
            while d.discovery_read.busy and time.monotonic()<deadline: APP.processEvents(); time.sleep(.005)
            APP.processEvents(); self.assertEqual(read.call_count,1); self.assertFalse(d.isVisible()); self.assertGreater(len(beats),2)
        timer.stop()


if __name__=='__main__': unittest.main()
