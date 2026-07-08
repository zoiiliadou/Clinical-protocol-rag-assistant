# Evaluation Report

## Project

Offline Medical RAG Q&A System

## Evaluation Goal

The goal of this evaluation is to verify whether the local RAG system can retrieve relevant evidence from the clinical protocol and generate concise answers grounded in the source document.

The evaluation focuses on four core clinical-trial question types:

1. Pregnancy-related exclusion criteria
2. Participant inclusion criteria
3. Study design
4. Dosing schedule

## Evaluation Setup

The system was evaluated using a small manually defined test set stored in:

```text
evaluation/eval_questions.json
