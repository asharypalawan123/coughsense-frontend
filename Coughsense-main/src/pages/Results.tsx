import { useLocation, useNavigate } from "react-router-dom";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Droplets, Wind, ArrowLeft, RotateCcw, Trash2 } from "lucide-react";
import { PredictionResult } from "./Index";
import { useState, useEffect } from "react";

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const result = location.state?.result as PredictionResult | undefined;
  
  const [history, setHistory] = useState<PredictionResult[]>([]);

  useEffect(() => {
    const stored = JSON.parse(localStorage.getItem('coughHistory') || '[]');
    setHistory(stored);
  }, []);

  const clearHistory = () => {
    localStorage.removeItem('coughHistory');
    setHistory([]);
  };

  const isDry = result?.predicted_cough_type === 'dry';

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30 flex flex-col">
      {/* Header */}
      <header className="py-4 px-4 flex items-center gap-3">
        <Button variant="ghost" size="icon" onClick={() => navigate('/')} className="rounded-full">
          <ArrowLeft className="w-5 h-5" />
        </Button>
        <h1 className="text-xl font-bold text-primary">CoughSense</h1>
      </header>

      {/* Main Content */}
      <main className="flex-1 px-4 pb-8 space-y-6">
        {/* Current Result */}
        {result && (
          <Card className="p-6 shadow-lg border-primary/10 animate-fade-in">
            <div className="text-center mb-6">
              <div className={`inline-flex items-center justify-center w-16 h-16 rounded-full mb-3 ${
                isDry ? 'bg-warning/10' : 'bg-info/20'
              }`}>
                {isDry ? (
                  <Wind className="w-8 h-8 text-warning" />
                ) : (
                  <Droplets className="w-8 h-8 text-info" />
                )}
              </div>
              
              <h2 className="text-2xl font-bold capitalize mb-1">
                {result.predicted_cough_type} Cough
              </h2>
              
              <p className="text-muted-foreground text-sm">
                {isDry 
                  ? "No mucus produced. Often caused by irritation or viral infections."
                  : "Produces mucus/phlegm. Usually indicates respiratory infection."
                }
              </p>
            </div>

            <div className="space-y-3 mb-6">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Confidence</span>
                <span className="font-bold">{result.confidence_score.toFixed(1)}%</span>
              </div>
              <Progress value={result.confidence_score} className="h-2" />
            </div>

            <Button onClick={() => navigate('/')} size="lg" className="w-full rounded-xl">
              <RotateCcw className="mr-2 w-5 h-5" />
              Analyze Another
            </Button>
          </Card>
        )}

        {!result && (
          <Card className="p-6 text-center">
            <p className="text-muted-foreground mb-4">No analysis result available</p>
            <Button onClick={() => navigate('/')}>Go to Home</Button>
          </Card>
        )}

        {/* History */}
        {history.length > 0 && (
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <h3 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide">
                Recent History
              </h3>
              <Button variant="ghost" size="sm" onClick={clearHistory} className="text-destructive hover:text-destructive">
                <Trash2 className="w-4 h-4 mr-1" />
                Clear
              </Button>
            </div>
            
            <div className="space-y-2">
              {history.slice(0, 5).map((item) => (
                <Card key={item.id} className="p-3 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                      item.predicted_cough_type === 'dry' ? 'bg-warning/10' : 'bg-info/20'
                    }`}>
                      {item.predicted_cough_type === 'dry' ? (
                        <Wind className="w-4 h-4 text-warning" />
                      ) : (
                        <Droplets className="w-4 h-4 text-info" />
                      )}
                    </div>
                    <div>
                      <p className="font-medium text-sm capitalize">{item.predicted_cough_type}</p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(item.timestamp).toLocaleDateString()} {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </p>
                    </div>
                  </div>
                  <span className="text-sm font-semibold">{item.confidence_score.toFixed(0)}%</span>
                </Card>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default Results;
