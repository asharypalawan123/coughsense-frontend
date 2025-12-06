import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Mic, Upload, AlertCircle, Square, Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { WaveformVisualizer } from "@/components/WaveformVisualizer";
import { useRef, useEffect } from "react";

export interface PredictionResult {
  predicted_cough_type: "dry" | "wet";
  confidence_score: number;
  timestamp: string;
  id: string;
}

type RecordingStatus = "idle" | "recording" | "processing" | "uploading";

const Index = () => {
  const [status, setStatus] = useState<RecordingStatus>("idle");
  const [audioLevel, setAudioLevel] = useState(0);
  const [recordingTime, setRecordingTime] = useState(0);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  
  const { toast } = useToast();
  const navigate = useNavigate();

  useEffect(() => {
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
      if (audioContextRef.current) audioContextRef.current.close();
    };
  }, []);

  const analyzeAudioLevel = () => {
    if (!analyserRef.current) return;
    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(dataArray);
    const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
    setAudioLevel(average / 255);
    animationFrameRef.current = requestAnimationFrame(analyzeAudioLevel);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 44100 } 
      });
      
      audioContextRef.current = new AudioContext();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      source.connect(analyserRef.current);
      analyzeAudioLevel();
      
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      chunksRef.current = [];
      
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        setStatus("processing");
        processAudio(blob);
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setStatus("recording");
      setRecordingTime(0);
      
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => {
          if (prev >= 5) {
            stopRecording();
            return 5;
          }
          return prev + 0.1;
        });
      }, 100);
      
      toast({ title: "Recording Started", description: "Cough clearly into your microphone" });
    } catch (error) {
      console.error("Error accessing microphone:", error);
      toast({ title: "Microphone Access Denied", description: "Please allow microphone access", variant: "destructive" });
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  const processAudio = async (blob: Blob) => {
    if (recordingTime < 0.5) {
      toast({ title: "Recording Too Short", description: "Please record for at least 1 second", variant: "destructive" });
      setStatus("idle");
      return;
    }

    setStatus("uploading");
    
    try {
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      
      reader.onloadend = async () => {
        const base64Audio = reader.result as string;
        const base64Data = base64Audio.split(',')[1];
        
        const { data, error } = await supabase.functions.invoke('predict-cough', {
          body: { audio: base64Data }
        });

        if (error) throw error;

        const result: PredictionResult = {
          predicted_cough_type: data.predicted_cough_type,
          confidence_score: data.confidence_score,
          timestamp: new Date().toISOString(),
          id: crypto.randomUUID(),
        };

        const history = JSON.parse(localStorage.getItem('coughHistory') || '[]');
        history.unshift(result);
        localStorage.setItem('coughHistory', JSON.stringify(history.slice(0, 20)));

        // Navigate to results page with result data
        navigate('/results', { state: { result } });
      };
    } catch (error: any) {
      console.error("Prediction error:", error);
      toast({ title: "Analysis Failed", description: error.message || "Please try again", variant: "destructive" });
      setStatus("idle");
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.includes('audio')) {
      toast({ title: "Invalid File", description: "Please upload an audio file (WAV or MP3)", variant: "destructive" });
      return;
    }

    setStatus("processing");
    processAudio(file);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30 flex flex-col">
      {/* Header */}
      <header className="py-6 px-4 text-center">
        <h1 className="text-3xl md:text-4xl font-bold text-primary">
          CoughSense
        </h1>
        <p className="text-muted-foreground mt-1 text-sm">
          Dry or Wet Cough Classifier
        </p>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex items-center justify-center px-4 pb-8">
        <Card className="w-full max-w-md p-6 md:p-8 shadow-lg border-primary/10">
          
          {status === "idle" && (
            <div className="space-y-6 animate-fade-in">
              <div className="text-center mb-4">
                <h2 className="text-xl font-semibold mb-1">Record Your Cough</h2>
                <p className="text-muted-foreground text-sm">3-5 seconds in a quiet space</p>
              </div>

              <div className="space-y-3">
                <Button
                  onClick={startRecording}
                  size="lg"
                  className="w-full h-16 bg-primary hover:bg-primary/90 flex gap-3 text-base rounded-xl"
                >
                  <Mic className="w-6 h-6" />
                  <span>Record Cough</span>
                </Button>
                
                <label className="cursor-pointer block">
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                  <Button
                    asChild
                    size="lg"
                    variant="outline"
                    className="w-full h-12 border-2 border-primary/20 rounded-xl"
                  >
                    <div className="flex gap-2">
                      <Upload className="w-5 h-5" />
                      <span>Upload Audio</span>
                    </div>
                  </Button>
                </label>
              </div>

              <div className="bg-muted/50 rounded-lg p-3 text-sm">
                <div className="flex gap-2">
                  <AlertCircle className="w-4 h-4 text-primary flex-shrink-0 mt-0.5" />
                  <ul className="text-muted-foreground space-y-0.5 text-xs">
                    <li>• Quiet environment</li>
                    <li>• Clear, natural cough</li>
                    <li>• 3-5 seconds only</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {status === "recording" && (
            <div className="space-y-6 text-center animate-fade-in">
              <div className="relative">
                <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-destructive/10 animate-pulse">
                  <Mic className="w-12 h-12 text-destructive" />
                </div>
                <WaveformVisualizer audioLevel={audioLevel} isRecording={true} />
              </div>
              
              <div>
                <h3 className="text-xl font-bold mb-1">Recording...</h3>
                <p className="text-lg text-primary font-mono">{recordingTime.toFixed(1)}s / 5.0s</p>
              </div>

              <Button onClick={stopRecording} variant="destructive" size="lg" className="rounded-xl">
                <Square className="mr-2 w-5 h-5" />
                Stop Recording
              </Button>
            </div>
          )}

          {(status === "processing" || status === "uploading") && (
            <div className="space-y-4 text-center animate-fade-in py-8">
              <Loader2 className="w-12 h-12 text-primary animate-spin mx-auto" />
              <div>
                <h3 className="text-xl font-semibold mb-1">Analyzing...</h3>
                <p className="text-muted-foreground text-sm">
                  {status === "processing" ? "Processing audio" : "Running prediction"}
                </p>
              </div>
            </div>
          )}
        </Card>
      </main>
    </div>
  );
};

export default Index;
