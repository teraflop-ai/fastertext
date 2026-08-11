use fasttext::FastText;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList};
use rayon::prelude::*;
use ahash::HashMap;

#[pyclass(name = "FasterText")]
struct FastTextPy {
    model: FastText,
    label_dict: HashMap<String, i16>,
    reverse_label_dict: HashMap<i16, String>,
}

#[pyfunction]
fn load_model(path: &str) -> PyResult<FastTextPy> {
    let model = FastText::load_model(path).map_err(|e| PyException::new_err(e.to_string()))?;
    let (labels, _) = model.get_labels();
    Ok(FastTextPy {
        label_dict: labels.iter().enumerate().map(|(i, l)| (l.clone(), i as i16)).collect(),
        reverse_label_dict: labels.into_iter().enumerate().map(|(i, l)| (i as i16, l)).collect(),
        model,
    })
}

#[pymethods]
impl FastTextPy {
    #[pyo3(signature = (texts, k=1, threshold=-1.0))]
    fn batch<'py>(
        &self,
        texts: &Bound<'py, PyList>,
        k: usize,
        threshold: f32,
    ) -> PyResult<(Bound<'py, PyArray2<i16>>, Bound<'py, PyArray2<f32>>)> {
        let py = texts.py();
        let texts: Vec<Option<String>> = texts.iter().map(|t| t.extract().ok()).collect();
        let rows: Vec<(Vec<i16>, Vec<f32>)> = py.detach(|| {
            texts
                .par_iter()
                .map(|t| {
                    let (mut l, mut p): (Vec<i16>, Vec<f32>) = t
                        .as_deref()
                        .map(|s| {
                            self.model
                                .predict(s, k, threshold)
                                .into_iter()
                                .map(|p| (*self.label_dict.get(&p.label).unwrap_or(&-1), p.prob))
                                .unzip()
                        })
                        .unwrap_or_default();
                    l.resize(k, 0);
                    p.resize(k, 0.0);
                    (l, p)
                })
                .collect()
        });
        let n = rows.len();
        let (mut labels, mut probs) = (Vec::with_capacity(n * k), Vec::with_capacity(n * k));
        for (l, p) in rows {
            labels.extend(l);
            probs.extend(p);
        }
        Ok((
            PyArray1::from_vec(py, labels).reshape([n, k])?,
            PyArray1::from_vec(py, probs).reshape([n, k])?,
        ))
    }

    fn get_labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.reverse_label_dict.clone().into_py_dict(py)
    }

    fn get_label_by_id(&self, id: i16) -> Option<String> {
        self.reverse_label_dict.get(&id).cloned()
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_class::<FastTextPy>()?;
    Ok(())
}