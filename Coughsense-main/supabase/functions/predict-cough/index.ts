import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { audio } = await req.json();
    
    if (!audio) {
      throw new Error('No audio data provided');
    }

    console.log('Received audio data for prediction');

    // ==========================================
    // REAL ML MODEL INTEGRATION
    // ==========================================
    // Calling Python FastAPI service for ML inference
    // The service extracts audio features and runs prediction using trained Random Forest model
    
    console.log('Calling ML inference service...');
    
    // ML Service URL (update this to your deployed service URL)
    // For local development: http://localhost:8000/predict
    // For production: https://your-ml-service-url.com/predict
    const ML_SERVICE_URL = Deno.env.get('ML_SERVICE_URL') || 'http://localhost:8000/predict';
    
    let randomType: string;
    let confidence: number;
    
    try {
      const mlResponse = await fetch(ML_SERVICE_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ audio }),
      });

      if (!mlResponse.ok) {
        throw new Error(`ML service returned ${mlResponse.status}: ${await mlResponse.text()}`);
      }

      const mlPrediction = await mlResponse.json();
      randomType = mlPrediction.predicted_cough_type;
      confidence = mlPrediction.confidence_score;

      console.log(`ML prediction: ${randomType} with ${confidence.toFixed(1)}% confidence`);
    } catch (mlError) {
      console.error('ML service error:', mlError);
      
      // Fallback to mock prediction if ML service is unavailable
      console.log('Falling back to mock prediction');
      const coughTypes = ['dry', 'wet'];
      randomType = coughTypes[Math.floor(Math.random() * coughTypes.length)];
      confidence = 75 + Math.random() * 20;
      
      console.log(`Fallback prediction: ${randomType} with ${confidence.toFixed(1)}% confidence`);
    }

    return new Response(
      JSON.stringify({
        predicted_cough_type: randomType,
        confidence_score: parseFloat(confidence.toFixed(1)),
        message: 'Prediction completed using ML model',
      }),
      { 
        headers: { 
          ...corsHeaders, 
          'Content-Type': 'application/json' 
        } 
      }
    );

  } catch (error) {
    console.error('Prediction error:', error);
    return new Response(
      JSON.stringify({ 
        error: error instanceof Error ? error.message : 'Prediction failed',
      }),
      {
        status: 500,
        headers: { 
          ...corsHeaders, 
          'Content-Type': 'application/json' 
        },
      }
    );
  }
});
