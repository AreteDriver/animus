pub mod graph;
pub mod matching;
pub mod models;
pub mod stability;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::graph::IntentGraph;
use crate::models::*;
use crate::stability::StabilityScorer;

/// Python-facing wrapper for IntentGraph
#[pyclass(name = "IntentGraph", unsendable)]
struct PyIntentGraph {
    inner: IntentGraph,
}

#[pymethods]
impl PyIntentGraph {
    #[new]
    #[pyo3(signature = (path=None))]
    fn new(path: Option<&str>) -> PyResult<Self> {
        let inner = match path {
            Some(p) => IntentGraph::persistent(p),
            None => IntentGraph::in_memory(),
        };
        inner
            .map(|g| PyIntentGraph { inner: g })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Publish an intent node to the graph. Returns computed stability.
    fn publish(&self, intent_dict: &Bound<'_, PyDict>) -> PyResult<f64> {
        let intent = dict_to_intent(intent_dict)?;
        self.inner
            .publish(&intent)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Query all intents, optionally filtered by minimum stability.
    #[pyo3(signature = (min_stability=None))]
    fn query_all(&self, py: Python, min_stability: Option<f64>) -> PyResult<Py<PyAny>> {
        let intents = self
            .inner
            .query_all(min_stability)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let list = PyList::empty(py);
        for intent in intents {
            list.append(intent_to_dict(py, &intent)?)?;
        }
        Ok(list.into())
    }

    /// Query intents from a specific agent.
    fn query_by_agent(&self, py: Python, agent_id: &str) -> PyResult<Py<PyAny>> {
        let intents = self
            .inner
            .query_by_agent(agent_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let list = PyList::empty(py);
        for intent in intents {
            list.append(intent_to_dict(py, &intent)?)?;
        }
        Ok(list.into())
    }

    /// Find overlapping intents for the given interface specs.
    fn find_overlapping(
        &self,
        py: Python,
        specs_list: &Bound<'_, PyList>,
        exclude_agent: &str,
        min_stability: f64,
    ) -> PyResult<Py<PyAny>> {
        let specs = list_to_interface_specs(specs_list)?;
        let intents = self
            .inner
            .find_overlapping(&specs, exclude_agent, min_stability)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let list = PyList::empty(py);
        for intent in intents {
            list.append(intent_to_dict(py, &intent)?)?;
        }
        Ok(list.into())
    }

    /// Resolve an intent against the graph. Returns adjustments and conflicts.
    fn resolve(
        &self,
        py: Python,
        intent_dict: &Bound<'_, PyDict>,
        min_stability: f64,
    ) -> PyResult<Py<PyAny>> {
        let intent = dict_to_intent(intent_dict)?;
        let result = self
            .inner
            .resolve(&intent, min_stability)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("original_intent", &result.original_intent)?;
        dict.set_item("is_clean", result.is_clean())?;
        dict.set_item("has_adjustments", result.has_adjustments())?;

        let adj_list = PyList::empty(py);
        for adj in &result.adjustments {
            let d = PyDict::new(py);
            d.set_item("kind", format!("{:?}", adj.kind))?;
            d.set_item("description", &adj.description)?;
            d.set_item("source_intent_id", &adj.source_intent_id)?;
            adj_list.append(d)?;
        }
        dict.set_item("adjustments", adj_list)?;

        let conflict_list = PyList::empty(py);
        for conflict in &result.conflicts {
            let d = PyDict::new(py);
            d.set_item("my_intent_id", &conflict.my_intent_id)?;
            d.set_item("their_intent_id", &conflict.their_intent_id)?;
            d.set_item("description", &conflict.description)?;
            d.set_item("their_stability", conflict.their_stability)?;
            d.set_item("resolution_suggestion", &conflict.resolution_suggestion)?;
            conflict_list.append(d)?;
        }
        dict.set_item("conflicts", conflict_list)?;

        let constraint_list = PyList::empty(py);
        for c in &result.adopted_constraints {
            let d = PyDict::new(py);
            d.set_item("target", &c.target)?;
            d.set_item("requirement", &c.requirement)?;
            constraint_list.append(d)?;
        }
        dict.set_item("adopted_constraints", constraint_list)?;

        Ok(dict.into())
    }

    /// Get intent count.
    fn count(&self) -> PyResult<usize> {
        self.inner
            .count()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get graph summary.
    fn summary(&self, py: Python) -> PyResult<Py<PyAny>> {
        let s = self
            .inner
            .summary()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("total_intents", s.total_intents)?;
        dict.set_item("agent_count", s.agent_count)?;
        dict.set_item("agents", s.agents)?;
        dict.set_item("average_stability", s.average_stability)?;
        dict.set_item("high_stability_count", s.high_stability_count)?;
        Ok(dict.into())
    }
}

/// Python-facing stability scorer
#[pyclass(name = "StabilityScorer")]
struct PyStabilityScorer {
    inner: StabilityScorer,
}

#[pymethods]
impl PyStabilityScorer {
    #[new]
    fn new() -> Self {
        PyStabilityScorer {
            inner: StabilityScorer::new(),
        }
    }

    fn compute(&self, intent_dict: &Bound<'_, PyDict>) -> PyResult<f64> {
        let intent = dict_to_intent(intent_dict)?;
        Ok(self.inner.compute(&intent))
    }
}

// ── Conversion helpers ──────────────────────────────────────────────

fn dict_to_intent(dict: &Bound<'_, PyDict>) -> PyResult<IntentNode> {
    let agent_id: String = dict
        .get_item("agent_id")?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("agent_id"))?
        .extract()?;
    let intent_text: String = dict
        .get_item("intent")?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("intent"))?
        .extract()?;

    let mut intent = IntentNode::new(&agent_id, &intent_text);

    if let Some(id) = dict.get_item("id")? {
        intent.id = id.extract()?;
    }

    if let Some(provides) = dict.get_item("provides")? {
        let list: &Bound<'_, PyList> = provides.cast()?;
        intent.provides = list_to_interface_specs(list)?;
    }

    if let Some(requires) = dict.get_item("requires")? {
        let list: &Bound<'_, PyList> = requires.cast()?;
        intent.requires = list_to_interface_specs(list)?;
    }

    if let Some(constraints) = dict.get_item("constraints")? {
        let list: &Bound<'_, PyList> = constraints.cast()?;
        intent.constraints = list_to_constraints(list)?;
    }

    if let Some(stability) = dict.get_item("stability")? {
        intent.stability = stability.extract()?;
    }

    if let Some(evidence) = dict.get_item("evidence")? {
        let list: &Bound<'_, PyList> = evidence.cast()?;
        intent.evidence = list_to_evidence(list)?;
    }

    if let Some(parent_id) = dict.get_item("parent_id")? {
        if !parent_id.is_none() {
            intent.parent_id = Some(parent_id.extract()?);
        }
    }

    Ok(intent)
}

fn list_to_interface_specs(list: &Bound<'_, PyList>) -> PyResult<Vec<InterfaceSpec>> {
    let mut specs = Vec::new();
    for item in list.iter() {
        let dict: &Bound<'_, PyDict> = item.cast()?;
        let name: String = dict
            .get_item("name")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("name"))?
            .extract()?;
        let kind_str: String = dict
            .get_item("kind")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("kind"))?
            .extract()?;
        let signature: String = dict
            .get_item("signature")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("signature"))?
            .extract()?;

        let kind = match kind_str.as_str() {
            "function" => InterfaceKind::Function,
            "class" => InterfaceKind::Class,
            "model" => InterfaceKind::Model,
            "endpoint" => InterfaceKind::Endpoint,
            "migration" => InterfaceKind::Migration,
            "config" => InterfaceKind::Config,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown InterfaceKind: '{}'. Expected one of: function, class, model, endpoint, migration, config",
                    other
                )));
            }
        };

        let mut spec = InterfaceSpec::new(&name, kind, &signature);

        if let Some(module) = dict.get_item("module_path")? {
            spec = spec.with_module(&module.extract::<String>()?);
        }

        if let Some(tags) = dict.get_item("tags")? {
            let tag_list: Vec<String> = tags.extract()?;
            let tag_refs: Vec<&str> = tag_list.iter().map(|s| s.as_str()).collect();
            spec = spec.with_tags(tag_refs);
        }

        specs.push(spec);
    }
    Ok(specs)
}

fn list_to_constraints(list: &Bound<'_, PyList>) -> PyResult<Vec<Constraint>> {
    let mut constraints = Vec::new();
    for item in list.iter() {
        let dict: &Bound<'_, PyDict> = item.cast()?;
        let target: String = dict
            .get_item("target")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("target"))?
            .extract()?;
        let requirement: String = dict
            .get_item("requirement")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("requirement"))?
            .extract()?;

        let mut constraint = Constraint::new(&target, &requirement);

        if let Some(affects) = dict.get_item("affects_tags")? {
            let tag_list: Vec<String> = affects.extract()?;
            let tag_refs: Vec<&str> = tag_list.iter().map(|s| s.as_str()).collect();
            constraint = constraint.with_affects(tag_refs);
        }

        constraints.push(constraint);
    }
    Ok(constraints)
}

fn list_to_evidence(list: &Bound<'_, PyList>) -> PyResult<Vec<Evidence>> {
    let mut evidence = Vec::new();
    for item in list.iter() {
        let dict: &Bound<'_, PyDict> = item.cast()?;
        let kind_str: String = dict
            .get_item("kind")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("kind"))?
            .extract()?;
        let description: String = dict
            .get_item("description")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("description"))?
            .extract()?;

        let ev = match kind_str.as_str() {
            "test_pass" => Evidence::test_pass(&description),
            "test_fail" => Evidence::conflict(&description), // test_fail treated as negative evidence
            "code_committed" => Evidence::code_committed(&description),
            "consumed_by" => Evidence::consumed_by(&description),
            "conflict" => Evidence::conflict(&description),
            "manual_approval" => Evidence::manual_approval(),
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown EvidenceKind: '{}'. Expected one of: test_pass, test_fail, code_committed, consumed_by, conflict, manual_approval",
                    other
                )));
            }
        };
        evidence.push(ev);
    }
    Ok(evidence)
}

fn intent_to_dict<'py>(py: Python<'py>, intent: &IntentNode) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &intent.id)?;
    dict.set_item("agent_id", &intent.agent_id)?;
    dict.set_item("timestamp", intent.timestamp.to_rfc3339())?;
    dict.set_item("intent", &intent.intent)?;
    dict.set_item("stability", intent.stability)?;
    dict.set_item("parent_id", &intent.parent_id)?;

    // Serialize provides
    let provides = PyList::empty(py);
    for spec in &intent.provides {
        let d = PyDict::new(py);
        d.set_item("name", &spec.name)?;
        d.set_item("kind", format!("{:?}", spec.kind))?;
        d.set_item("signature", &spec.signature)?;
        d.set_item("module_path", &spec.module_path)?;
        d.set_item("tags", &spec.tags)?;
        provides.append(d)?;
    }
    dict.set_item("provides", provides)?;

    // Serialize requires
    let requires = PyList::empty(py);
    for spec in &intent.requires {
        let d = PyDict::new(py);
        d.set_item("name", &spec.name)?;
        d.set_item("kind", format!("{:?}", spec.kind))?;
        d.set_item("signature", &spec.signature)?;
        d.set_item("module_path", &spec.module_path)?;
        d.set_item("tags", &spec.tags)?;
        requires.append(d)?;
    }
    dict.set_item("requires", requires)?;

    // Serialize constraints
    let constraints = PyList::empty(py);
    for c in &intent.constraints {
        let d = PyDict::new(py);
        d.set_item("target", &c.target)?;
        d.set_item("requirement", &c.requirement)?;
        d.set_item("affects_tags", &c.affects_tags)?;
        constraints.append(d)?;
    }
    dict.set_item("constraints", constraints)?;

    Ok(dict)
}

/// Python module definition
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIntentGraph>()?;
    m.add_class::<PyStabilityScorer>()?;
    Ok(())
}
