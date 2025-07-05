# AI Symptom Checker - Advanced Medical Analysis

A comprehensive AI-powered symptom analysis system with advanced prompting, explainability, role-based access control, and real-time analytics.

## 🚀 Features

### Core Functionality

- **Advanced Symptom Analysis**: AI-powered analysis using GPT-3.5-turbo with comprehensive medical insights
- **Multiple Condition Identification**: Always provides ≥3 possible conditions with confidence levels
- **Urgency Assessment**: Intelligent urgency classification (low/medium/high/emergency)
- **Red Flag Detection**: Automatic identification of serious symptoms requiring immediate attention
- **Explainability**: Clear reasoning for each suggested condition
- **Follow-up Questions**: Contextual questions to gather more information

### Advanced Prompting

- **Structured Analysis**: Demands multiple conditions with urgency weighting
- **Evidence-based Reasoning**: Explains why each condition is suggested
- **Differential Diagnosis**: Considers multiple body systems when appropriate
- **Medical Safety**: Prioritizes patient safety with red flag detection
- **ICD-10 Codes**: Includes medical coding for conditions

### Role-Based System

- **Public Access**: Rate-limited symptom analysis for general users
- **Admin Panel**: Comprehensive analytics and system management
- **Query Tracking**: Detailed logging of all analysis requests
- **System Monitoring**: Real-time health checks and performance metrics

### Enhanced UI/UX

- **Chat-style Interface**: Modern, responsive design with real-time interaction
- **Confidence Visualization**: Visual indicators for analysis confidence
- **Conditional Coloring**: Color-coded urgency and risk levels
- **Real-time Updates**: Live status indicators and typing animations
- **Mobile Responsive**: Optimized for all device sizes

### Metrics & Analytics

- **Comprehensive Logging**: Tracks symptoms, processing time, model usage, and outcomes
- **Privacy-safe Analytics**: Anonymized data collection for insights
- **Performance Monitoring**: Real-time system health and performance metrics
- **Trend Analysis**: Historical data analysis and pattern recognition

## 🏗️ Architecture

```
ai-symptom-checker/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── symptoms.py      # Main symptom analysis API
│   │   │   └── admin.py         # Admin panel API
│   │   ├── models/
│   │   │   └── symptom.py       # Pydantic models with explainability
│   │   ├── services/
│   │   │   └── llm_service.py   # Advanced LLM service with metrics
│   │   ├── config.py            # Configuration with admin settings
│   │   └── main.py              # FastAPI application
│   ├── requirements.txt
│   └── .env                     # Environment variables
├── frontend/
│   ├── index.html               # Main chat interface
│   └── admin.html               # Admin dashboard
└── README.md
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- Redis (optional, for caching and analytics)

### Backend Setup

1. **Clone and navigate to backend:**

```bash
cd backend
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Configure environment variables:**

```bash
# Create .env file
OPENAI_API_KEY=your_openai_api_key_here
ADMIN_API_KEY=your_secure_admin_key_here
REDIS_URL=redis://localhost:6379  # Optional
ENVIRONMENT=PRODUCTION
```

4. **Start the server:**

```bash
PYTHONPATH=/path/to/backend uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. **Open the main interface:**

   - Navigate to `frontend/index.html` in your browser
   - Or serve it with a local server

2. **Access the admin dashboard:**
   - Open `frontend/admin.html` in your browser
   - Use your admin API key to connect

## 📊 API Endpoints

### Public Endpoints

- `POST /api/v1/analyze` - Analyze symptoms
- `GET /api/v1/health` - Health check
- `GET /api/v1/info` - API information

### Admin Endpoints (require API key)

- `GET /api/v1/admin/metrics` - Get analytics data
- `GET /api/v1/admin/recent-queries` - Get recent queries
- `GET /api/v1/admin/system-status` - System health
- `DELETE /api/v1/admin/clear-cache` - Clear cache
- `GET /api/v1/admin/health-check` - Admin health check

## 🎯 Usage Examples

### Basic Symptom Analysis

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": "fever, headache, and fatigue for 3 days",
    "age": 35,
    "gender": "male",
    "medical_history": "diabetes"
  }'
```

### Admin Analytics

```bash
curl -X GET "http://localhost:8000/api/v1/admin/metrics?days=7&api_key=your_admin_key"
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `ADMIN_API_KEY`: Secure key for admin access
- `REDIS_URL`: Redis connection URL (optional)
- `ENVIRONMENT`: Production/Development environment
- `OPENAI_MODEL`: AI model to use (default: gpt-3.5-turbo)

### Rate Limiting

- Default: 60 requests per minute per IP
- Configurable via `rate_limit_per_minute` setting

### Caching

- Redis-based caching for analysis results
- Configurable TTL (default: 1 hour)
- Automatic cache invalidation

## 📈 Analytics Features

### Metrics Tracked

- **Request Analytics**: Symptoms, processing time, model usage
- **Performance Metrics**: Response times, cache hit rates
- **Medical Insights**: Urgency distribution, red flag frequency
- **User Patterns**: Common symptoms, confidence trends

### Admin Dashboard Features

- **Real-time Charts**: Urgency distribution and top symptoms
- **Query History**: Recent analysis requests with details
- **System Status**: Health monitoring and performance metrics
- **Cache Management**: Clear cache and view cache statistics

## 🔒 Security & Privacy

### Data Protection

- **Anonymized Logging**: No personal information stored
- **Secure API Keys**: Environment-based configuration
- **Rate Limiting**: Prevents abuse and ensures fair usage
- **Input Validation**: Comprehensive request validation

### Access Control

- **Public API**: Rate-limited access for symptom analysis
- **Admin Access**: Secure API key-based admin panel
- **CORS Configuration**: Configurable cross-origin settings

## 🚨 Medical Disclaimer

**IMPORTANT**: This system is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for proper medical care.




