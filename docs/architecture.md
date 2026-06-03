# Architecture

> Polished architecture writeup + diagram. (Phase 10 fills this from the source
> design in `../customer_care_bot_architecture.txt` and `../EXECUTION_PLAN.md`.)

## 1. Core idea
_TODO: the resolve-don't-tell thesis + the five differentiators._

## 2. High-level layers
_TODO: client → orchestration (agent loop) → knowledge/decision/tools/memory → stores. Diagram._

## 3. Tech stack
_TODO: Next.js · FastAPI · Gemini Flash (google-genai) · LightRAG · Neo4j · SQLite/Supabase._

## 4. The agent loop
_TODO: PERCEIVE → RETRIEVE → DECIDE → ACT → RESPOND → REMEMBER; 2 LLM calls/turn._

## 5. Data model
_TODO: Neo4j graph schema · document corpus · memory tables · mock business DB._

## 6. Tools / actions
_TODO: function-calling schemas, confirmation gates._

## 7. Guardrails, security, scale
_TODO: grounding, PII masking, error fallbacks, stateless/horizontal scale._
