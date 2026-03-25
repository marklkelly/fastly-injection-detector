//! Fastly Compute entrypoint for the injection detector.
//!
//! This service exposes two endpoints:
//!
//! * `GET /health` – returns "ok" for health checks.
//! * `POST /classify` – expects a JSON payload with a `text` field.  It
//!   returns a JSON response containing the predicted label, per‑class
//!   probabilities and timing metrics.  See `README.md` for example usage.

use fastly::http::{Method, StatusCode};
use fastly::{Request, Response};
use serde_json::json;

mod classify_tract_simple;

#[fastly::main]
fn main(mut req: Request) -> Result<Response, fastly::Error> {
    match (req.get_method(), req.get_path()) {
        (&Method::GET, "/health") => {
            Ok(Response::from_status(StatusCode::OK).with_body_text_plain("ok"))
        }
        (&Method::POST, "/classify") => {
            // Enforce Content-Type: application/json
            let ct_ok = req
                .get_header_str("content-type")
                .map(|ct| ct.starts_with("application/json"))
                .unwrap_or(false);
            if !ct_ok {
                return Ok(Response::from_status(StatusCode::UNSUPPORTED_MEDIA_TYPE)
                    .with_content_type(mime::APPLICATION_JSON)
                    .with_body(
                        json!({"error": "unsupported_media_type",
                                      "message": "Content-Type must be application/json"})
                        .to_string(),
                    ));
            }

            // Read body with size limit (64 KB)
            let body = req.take_body_str();
            if body.len() > 65_536 {
                return Ok(Response::from_status(StatusCode::PAYLOAD_TOO_LARGE)
                    .with_content_type(mime::APPLICATION_JSON)
                    .with_body(
                        json!({"error": "payload_too_large",
                                      "message": "Request body must not exceed 64 KB"})
                        .to_string(),
                    ));
            }

            // Parse JSON — no fallback to raw body
            let text: String = match serde_json::from_str::<serde_json::Value>(&body) {
                Ok(v) => match v.get("text").and_then(|t| t.as_str()) {
                    Some(t) => t.to_string(),
                    None => {
                        return Ok(Response::from_status(StatusCode::BAD_REQUEST)
                            .with_content_type(mime::APPLICATION_JSON)
                            .with_body(
                                json!({"error": "invalid_request",
                                             "message": "Missing required field: \"text\""})
                                .to_string(),
                            ));
                    }
                },
                Err(_) => {
                    return Ok(Response::from_status(StatusCode::BAD_REQUEST)
                        .with_content_type(mime::APPLICATION_JSON)
                        .with_body(
                            json!({"error": "invalid_json",
                                          "message": "Request body must be valid JSON"})
                            .to_string(),
                        ));
                }
            };

            let result = classify_tract_simple::classify(&text);

            match result {
                Ok(out) => {
                    let resp = json!({
                        "label": out.label,
                        "score": out.score,
                        "injection_score": out.injection_score,
                        "elapsed_ms": out.elapsed_ms,
                        "tokenization_ms": out.tokenization_ms,
                        "injection_inference_ms": out.injection_inference_ms,
                        "postprocess_ms": out.postprocess_ms,
                    })
                    .to_string();
                    Ok(Response::from_status(StatusCode::OK)
                        .with_content_type(mime::APPLICATION_JSON)
                        .with_body(resp))
                }
                Err(e) => {
                    let err_resp = json!({
                        "error": "classification_failed",
                        "message": e.to_string(),
                    })
                    .to_string();
                    Ok(Response::from_status(StatusCode::INTERNAL_SERVER_ERROR)
                        .with_content_type(mime::APPLICATION_JSON)
                        .with_body(err_resp))
                }
            }
        }
        _ => Ok(Response::from_status(StatusCode::NOT_FOUND)
            .with_content_type(mime::APPLICATION_JSON)
            .with_body(
                json!({"error": "not_found",
                                  "message": "The requested resource was not found"})
                .to_string(),
            )),
    }
}
