//! Structural semantic matching utilities — deterministic, no LLM.
//!
//! Provides normalization and comparison functions for interface names,
//! type signatures, and constraint targets. Mirrors the Python matching module.

/// Known suffixes to strip for name normalization.
const NAME_SUFFIXES: &[&str] = &[
    "Model",
    "Service",
    "Handler",
    "Controller",
    "Spec",
    "Interface",
];

/// Normalize an interface name for comparison.
///
/// Lowercase, strip known suffixes, split CamelCase into tokens.
pub fn normalize_name(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }

    // Strip known suffixes
    let mut stripped = name;
    for suffix in NAME_SUFFIXES {
        if stripped.ends_with(suffix) && stripped.len() > suffix.len() {
            stripped = &stripped[..stripped.len() - suffix.len()];
            break;
        }
    }

    // Split CamelCase into tokens
    let tokens = split_camel_case(stripped);
    if tokens.is_empty() {
        return stripped.to_lowercase();
    }

    tokens
        .iter()
        .map(|t| t.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Split a CamelCase string into tokens.
fn split_camel_case(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = s.chars().collect();
    for i in 0..chars.len() {
        let c = chars[i];
        if c.is_uppercase() && !current.is_empty() {
            let next_is_lower = i + 1 < chars.len() && chars[i + 1].is_lowercase();
            let prev_is_lower = current.chars().last().is_some_and(|p| p.is_lowercase());
            if prev_is_lower || next_is_lower {
                tokens.push(current.clone());
                current.clear();
            }
        }
        current.push(c);
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

/// Check if two names refer to the same concept.
///
/// Returns true if normalized names are equal, one is a prefix
/// of the other, or one contains the other.
pub fn names_overlap(a: &str, b: &str) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }

    let na = normalize_name(a);
    let nb = normalize_name(b);

    if na == nb {
        return true;
    }

    // Prefix match
    if na.starts_with(&*nb) || nb.starts_with(&*na) {
        return true;
    }

    // Containment match
    na.contains(&*nb) || nb.contains(&*na)
}

/// Normalize a type string for comparison.
///
/// Handles aliases (UUID<->uuid, String<->str, i64<->int),
/// Optional\[X\] -> X, list\[X\]<->Vec\<X\><->List\[X\].
pub fn normalize_type(t: &str) -> String {
    let t = t.trim();
    if t.is_empty() {
        return String::new();
    }

    let mut t = t.to_string();

    // Handle Optional[X] -> X
    if t.starts_with("Optional[") && t.ends_with(']') {
        t = t[9..t.len() - 1].to_string();
    }

    // Handle X | None or None | X
    if t.contains(" | ") {
        let parts: Vec<&str> = t
            .split(" | ")
            .map(|p| p.trim())
            .filter(|p| *p != "None")
            .collect();
        if let Some(first) = parts.first() {
            t = first.to_string();
        } else {
            return String::new();
        }
    }

    // Handle generic containers
    if let Some(inner) = extract_container_inner(&t) {
        let normalized_inner = normalize_type(&inner);
        return format!("list[{}]", normalized_inner);
    }

    // Direct alias lookup
    match t.as_str() {
        "UUID" | "uuid" => "uuid".to_string(),
        "str" | "String" | "string" => "str".to_string(),
        "int" | "i32" | "i64" | "i128" | "u32" | "u64" => "int".to_string(),
        "float" | "f32" | "f64" => "float".to_string(),
        "bool" | "boolean" => "bool".to_string(),
        _ => t.to_lowercase(),
    }
}

/// Extract the inner type from container types like list[X], List[X], Vec<X>.
fn extract_container_inner(t: &str) -> Option<String> {
    if (t.starts_with("list[") || t.starts_with("List[")) && t.ends_with(']') {
        return Some(t[5..t.len() - 1].trim().to_string());
    }
    if t.starts_with("Vec<") && t.ends_with('>') {
        return Some(t[4..t.len() - 1].trim().to_string());
    }
    None
}

/// Parse "field: type, field: type" into a vector of (field, type) pairs.
pub fn parse_signature(sig: &str) -> Vec<(String, String)> {
    if sig.trim().is_empty() {
        return Vec::new();
    }

    sig.split(',')
        .filter_map(|part| {
            let part = part.trim();
            part.split_once(':')
                .map(|(field, type_str)| (field.trim().to_string(), type_str.trim().to_string()))
        })
        .collect()
}

/// Check if signature b is compatible with signature a.
///
/// Compatible if b's fields are a superset of a's fields with normalized types.
/// Empty a is compatible with anything.
pub fn signatures_compatible(a: &str, b: &str) -> bool {
    let fields_a = parse_signature(a);
    let fields_b = parse_signature(b);

    if fields_a.is_empty() {
        return true;
    }

    for (field, type_a) in &fields_a {
        match fields_b.iter().find(|(f, _)| f == field) {
            Some((_, type_b)) => {
                if normalize_type(type_a) != normalize_type(type_b) {
                    return false;
                }
            }
            None => return false,
        }
    }

    true
}

/// Normalize a constraint target for comparison.
///
/// Lowercase, strip "model"/"service" suffix, replace
/// underscores/hyphens with spaces, collapse whitespace.
pub fn normalize_constraint_target(target: &str) -> String {
    if target.is_empty() {
        return String::new();
    }

    let mut t = target.to_lowercase();

    // Replace underscores and hyphens with spaces
    t = t.replace(['_', '-'], " ");

    // Collapse whitespace
    t = t.split_whitespace().collect::<Vec<_>>().join(" ");

    // Strip known suffixes
    for suffix in &["model", "service"] {
        let with_space = format!(" {}", suffix);
        if t.ends_with(&with_space) {
            t = t[..t.len() - with_space.len()].to_string();
        }
    }

    t.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_name_strip_suffix() {
        assert_eq!(normalize_name("UserModel"), "user");
        assert_eq!(normalize_name("AuthService"), "auth");
        assert_eq!(normalize_name("RequestHandler"), "request");
        assert_eq!(normalize_name("UserController"), "user");
        assert_eq!(normalize_name("ApiSpec"), "api");
        assert_eq!(normalize_name("DatabaseInterface"), "database");
    }

    #[test]
    fn test_normalize_name_camel_case() {
        assert_eq!(normalize_name("UserProfile"), "user profile");
        assert_eq!(normalize_name("MealPlanService"), "meal plan");
    }

    #[test]
    fn test_normalize_name_simple() {
        assert_eq!(normalize_name("User"), "user");
        assert_eq!(normalize_name("user"), "user");
    }

    #[test]
    fn test_normalize_name_empty() {
        assert_eq!(normalize_name(""), "");
    }

    #[test]
    fn test_names_overlap_exact_after_suffix() {
        assert!(names_overlap("UserModel", "User"));
        assert!(names_overlap("AuthService", "Auth"));
    }

    #[test]
    fn test_names_overlap_containment() {
        assert!(names_overlap("User", "UserProfile"));
    }

    #[test]
    fn test_names_overlap_no_match() {
        assert!(!names_overlap("User", "Recipe"));
        assert!(!names_overlap("AuthService", "RecipeService"));
    }

    #[test]
    fn test_names_overlap_empty() {
        assert!(!names_overlap("", "User"));
        assert!(!names_overlap("User", ""));
    }

    #[test]
    fn test_normalize_type_aliases() {
        assert_eq!(normalize_type("UUID"), "uuid");
        assert_eq!(normalize_type("String"), "str");
        assert_eq!(normalize_type("i64"), "int");
        assert_eq!(normalize_type("f64"), "float");
        assert_eq!(normalize_type("boolean"), "bool");
    }

    #[test]
    fn test_normalize_type_optional() {
        assert_eq!(normalize_type("Optional[str]"), "str");
        assert_eq!(normalize_type("str | None"), "str");
    }

    #[test]
    fn test_normalize_type_containers() {
        assert_eq!(normalize_type("Vec<String>"), "list[str]");
        assert_eq!(normalize_type("List[str]"), "list[str]");
        assert_eq!(normalize_type("list[str]"), "list[str]");
    }

    #[test]
    fn test_signatures_compatible_superset() {
        assert!(signatures_compatible(
            "id: UUID, email: str",
            "id: UUID, email: str, name: str"
        ));
    }

    #[test]
    fn test_signatures_compatible_type_alias() {
        assert!(signatures_compatible(
            "id: UUID, name: String",
            "id: uuid, name: str"
        ));
    }

    #[test]
    fn test_signatures_incompatible_missing_field() {
        assert!(!signatures_compatible("id: UUID, email: str", "id: UUID"));
    }

    #[test]
    fn test_signatures_incompatible_type_mismatch() {
        assert!(!signatures_compatible("id: UUID", "id: int"));
    }

    #[test]
    fn test_signatures_compatible_empty() {
        assert!(signatures_compatible("", "id: UUID"));
        assert!(signatures_compatible("", ""));
    }

    #[test]
    fn test_normalize_constraint_target() {
        assert_eq!(normalize_constraint_target("User Model"), "user");
        assert_eq!(normalize_constraint_target("user_model"), "user");
        assert_eq!(normalize_constraint_target("User model"), "user");
        assert_eq!(normalize_constraint_target("user-service"), "user");
    }

    #[test]
    fn test_normalize_constraint_target_no_suffix() {
        assert_eq!(
            normalize_constraint_target("authentication"),
            "authentication"
        );
        assert_eq!(normalize_constraint_target("User.id"), "user.id");
    }

    #[test]
    fn test_normalize_constraint_target_empty() {
        assert_eq!(normalize_constraint_target(""), "");
    }
}
