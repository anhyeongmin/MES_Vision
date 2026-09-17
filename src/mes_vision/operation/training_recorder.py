"""Record untouched camera RGB with capture timing and explicit clip labels."""
from datetime import datetime, timezone
from pathlib import Path
from fractions import Fraction
from uuid import uuid4
import json, os, shutil, time


class TrainingRecorder:
    def __init__(self, root, metadata, camera_settings):
        import av
        label = metadata['label']
        if label not in ('OK','NG01','NG02','NG03','NG04','NG05','NG06','MIXED','UNKNOWN'):
            raise ValueError('수집 분류 오류')
        if not metadata.get('product_id'): raise ValueError('품목을 선택하세요.')
        root = Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
        self.folder = root / (datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+label+'_'+uuid4().hex[:8])
        self.folder.mkdir(); self.path = self.folder / (label+'.mp4')
        self.meta = dict(schema_version=1, status='RECORDING', created_utc=datetime.now(timezone.utc).isoformat(),
            **metadata, camera_settings=camera_settings, preprocessing='raw_camera_rgb', annotations_reviewed=False,
            label_scope='clip_only_not_pixel_annotations', frames=0, video=self.path.name,
            encoding={'codec':'mpeg4','bit_rate':12000000,'pixel_format':'yuv420p','variable_frame_timestamps':True})
        self.container=None; self.times=None; self.start=None; self.last=None; self.last_pts=-1
        self.write_manifest()
        try:
            if shutil.disk_usage(root).free < 512*1024**2: raise OSError('녹화 저장 공간이 부족합니다.')
            self.container=av.open(str(self.path),'w')
            # Native FFmpeg encoder; do not introduce a libx264 encoder dependency.
            self.stream=self.container.add_stream('mpeg4',rate=camera_settings['fps'])
            self.stream.width=camera_settings['width']; self.stream.height=camera_settings['height']
            self.stream.pix_fmt='yuv420p'; self.stream.time_base=Fraction(1,60000)
            self.stream.codec_context.time_base=Fraction(1,60000)
            self.stream.codec_context.max_b_frames=0
            self.stream.bit_rate=12000000
            self.times=(self.folder/'frames.jsonl').open('x',encoding='utf-8')
        except Exception:
            self.close('FAILED'); raise

    def write_manifest(self):
        temporary=self.folder/'recording.tmp'
        with temporary.open('w',encoding='utf-8') as f:
            json.dump(self.meta,f,ensure_ascii=False,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
        temporary.replace(self.folder/'recording.json')

    def add(self, frame):
        import av
        now=time.monotonic()
        if self.meta['frames']%30 == 0 and shutil.disk_usage(self.folder).free < 256*1024**2:
            raise OSError('저장 공간 부족으로 녹화를 중단했습니다.')
        if (frame.width,frame.height)!=(self.stream.width,self.stream.height): raise ValueError('녹화 영상 크기가 변경됐습니다.')
        if self.last is not None and frame.sequence<=self.last: raise ValueError('녹화 프레임 순서 오류')
        self.start=now if self.start is None else self.start
        pts=max(self.last_pts+20,round((now-self.start)*1000000)) if self.last_pts>=0 else 0
        video=av.VideoFrame.from_ndarray(frame.rgb,format='rgb24'); video.pts=round(pts*.06); video.time_base=Fraction(1,60000)
        for packet in self.stream.encode(video): self.container.mux(packet)
        self.times.write(json.dumps({'sequence':frame.sequence,'pts_us':pts,'host_received_monotonic':now,
            'source_frame_id':frame.frame_id,'device_time_seconds':frame.media_time_seconds})+'\n')
        self.last=frame.sequence; self.last_pts=pts; self.meta['frames']+=1

    def close(self, status='COMPLETED'):
        error=None
        if self.container:
            try:
                for packet in self.stream.encode(): self.container.mux(packet)
            except Exception as exc: error=exc
            finally:
                try: self.container.close()
                except Exception as exc: error=exc
                self.container=None
        if self.times: self.times.close(); self.times=None
        self.meta.update(status='FAILED' if error else status, duration_seconds=max(0,self.last_pts)/1000000,
                         ended_utc=datetime.now(timezone.utc).isoformat())
        if error: self.meta['error']=str(error)
        self.write_manifest()
        if error: raise error
        return str(self.folder)
