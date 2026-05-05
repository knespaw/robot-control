use std::fmt::Debug;
use std::path::PathBuf;

use ort::compiler::ModelCompiler;
use ort::ep::coreml::{ComputeUnits, ModelFormat, SpecializationStrategy};
use ort::ep::{CoreML, ExecutionProviderDispatch};
use ort::memory::Allocator;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::{IoBinding, Session, SessionOutputs};
use ort::value::{PrimitiveTensorElementType, Tensor};
use tracing::{debug, info};

use super::utils::*;
use crate::utils::files::get_path;



const CACHE_DIR : &str = ".cache/models";
const MODELS_DIR : &str = "assets/models";



pub(crate) trait TensorType:
	Clone + Debug + Default + PrimitiveTensorElementType + 'static
{
}

impl<T> TensorType for T where T : Clone + Debug + Default + PrimitiveTensorElementType + 'static {}



pub(crate) struct TensorSymbol<T>
where
	T : TensorType,
{
	key :          &'static str,
	shape :        Vec<usize>,
	_placeholder : Option<Tensor<T>>,
}

impl<T> TensorSymbol<T>
where
	T : TensorType,
{
	pub(crate) fn new<V>(
		key : &'static str,
		shape : V,
	) -> MlResult<Self>
	where
		V : Into<Vec<usize>>,
	{
		let shape = shape.into();

		if shape.is_empty()
		{
			Err(MlError::Tensor("tensor must have a non-zero shape".into()))
		}
		else
		{
			let _placeholder = None;

			Ok(TensorSymbol { key, shape, _placeholder })
		}
	}

	pub(crate) fn initialize(
		&mut self,
		allocator : &Allocator,
	) -> MlResult<()>
	{
		self._placeholder = Some(
			Tensor::<T>::new(allocator, self.shape.clone())
				.map_err(|e| MlError::Tensor(format!("could not initialize tensor >> {}", e)))?,
		);

		Ok(())
	}

	pub(crate) unsafe fn tensor(&self) -> &Tensor<T>
	{
		unsafe {
			self._placeholder
				.as_ref()
				.unwrap_unchecked()
		}
	}

	pub(crate) unsafe fn held_data(&mut self) -> &mut [T]
	{
		unsafe {
			self._placeholder
				.as_mut()
				.unwrap_unchecked()
				.extract_tensor_mut()
		}
		.1
	}
}



pub(crate) struct ExecutionParameters
{
	static_input :         bool,
	enable_subgraphs :     bool,
	model_format :         ModelFormat,
	strategy :             SpecializationStrategy,
	compute_units :        ComputeUnits,
	profile_compute_plan : bool,
	low_fp_accumulation :  bool,
}

impl Default for ExecutionParameters
{
	fn default() -> Self
	{
		ExecutionParameters {
			static_input :         true,
			enable_subgraphs :     true,
			model_format :         ModelFormat::MLProgram,
			strategy :             SpecializationStrategy::FastPrediction,
			compute_units :        ComputeUnits::CPUAndNeuralEngine,
			profile_compute_plan : false,
			low_fp_accumulation :  false, //true,
		}
	}
}



pub(crate) struct InferenceParameters
{
	parallel_execution :     bool,
	/// Maximum number of threads to parallelize the execution of the graph.
	inter_threads :          Option<usize>,
	/// Maximum number of threads to parallelize execution within nodes.
	intra_threads :          Option<usize>,
	gelu_approximation :     bool,
	cast_chain_elimination : bool,
	flush_to_zero :          bool,
	/// Use memory pattern optimization (for static batches).
	mem_pattern_opt :        bool,
	/// Use slower, deterministic kernels for reproducible results.
	deterministic_compute :  bool,
	opt_level :              GraphOptimizationLevel,
}

impl Default for InferenceParameters
{
	fn default() -> Self
	{
		InferenceParameters {
			parallel_execution :     true,
			inter_threads :          None,
			intra_threads :          None,
			gelu_approximation :     false, //true,
			mem_pattern_opt :        true,
			deterministic_compute :  false,
			opt_level :              GraphOptimizationLevel::All,
			flush_to_zero :          false,
			cast_chain_elimination : false,
		}
	}
}



pub(crate) struct Engine
{
	session : Session,
	io :      IoBinding,
}

impl Engine
{
	pub(crate) fn start<T, V>(
		model_subpath : &str,
		execution_parameters : ExecutionParameters,
		inference_parameters : InferenceParameters,
		output_symbols : Vec<TensorSymbol<T>>,
		input_symbols : Vec<&mut TensorSymbol<V>>,
	) -> MlResult<Self>
	where
		T : TensorType,
		V : TensorType,
	{
		info!(model_subpath, "starting inference engine");

		let model_path = Self::get_model_full_path(model_subpath)?;

		let executor = Self::configure_execution(execution_parameters)?;

		let mut session_builder = Self::configure_computation(executor, inference_parameters)?;

		let session = session_builder
			.commit_from_file(&model_path)
			.map_err(|e| MlError::Init(e.to_string()))?;

		let mut io = session
			.create_binding()
			.map_err(MlError::Binding)?;

		Self::prebind_outputs(&mut io, &session, output_symbols)?;

		Self::prebind_inputs(&mut io, &session, input_symbols)?;

		info!(model_path, "inference engine ready");

		Ok(Engine { session, io })
	}

	fn prebind_outputs<T>(
		io : &mut IoBinding,
		session : &Session,
		output_symbols : Vec<TensorSymbol<T>>,
	) -> MlResult<()>
	where
		T : TensorType,
	{
		for mut output in output_symbols
		{
			output.initialize(session.allocator())?;

			io.bind_output(output.key, output._placeholder.unwrap())
				.map_err(MlError::Binding)?;
		}

		Ok(())
	}

	fn prebind_inputs<T>(
		io : &mut IoBinding,
		session : &Session,
		input_symbols : Vec<&mut TensorSymbol<T>>,
	) -> MlResult<()>
	where
		T : TensorType,
	{
		for input in input_symbols
		{
			input.initialize(session.allocator())?;

			io.bind_input(input.key, unsafe { input.tensor() })
				.map_err(MlError::Binding)?;
		}

		Ok(())
	}

	fn get_model_full_path(model_subpath : &str) -> MlResult<String>
	{
		let path =
			get_path(MODELS_DIR, model_subpath).map_err(|e| MlError::ModelFile(e.to_string()))?;

		if !path.ends_with(".onnx")
		{
			return Err(MlError::ModelFile("file must be in .onnx format".into()));
		}

		Ok(path)
	}

	fn configure_execution(
		execution_parameters : ExecutionParameters
	) -> MlResult<ExecutionProviderDispatch>
	{
		Ok(CoreML::default()
			.with_subgraphs(execution_parameters.enable_subgraphs)
			.with_compute_units(execution_parameters.compute_units)
			.with_low_precision_accumulation_on_gpu(execution_parameters.low_fp_accumulation)
			.with_profile_compute_plan(execution_parameters.profile_compute_plan)
			.with_specialization_strategy(execution_parameters.strategy)
			.with_static_input_shapes(execution_parameters.static_input)
			.with_model_format(execution_parameters.model_format)
			.with_model_cache_dir(
				get_path(CACHE_DIR, "").map_err(|e| MlError::Init(e.to_string()))?,
			)
			.build())
	}

	fn configure_computation(
		executor : ExecutionProviderDispatch,
		inference_parameters : InferenceParameters,
	) -> MlResult<SessionBuilder>
	{
		let mut session_builder = Session::builder()
			.map_err(|e| MlError::Init(e.to_string()))?
			.with_execution_providers([executor])
			.map_err(|e| MlError::Init(e.to_string()))?;

		session_builder = session_builder
			.with_parallel_execution(inference_parameters.parallel_execution)
			.map_err(|e| MlError::SessionBuild(e.to_string()))?;

		session_builder = session_builder
			.with_memory_pattern(inference_parameters.mem_pattern_opt)
			.map_err(|e| MlError::SessionBuild(e.to_string()))?;

		session_builder = session_builder
			.with_deterministic_compute(inference_parameters.deterministic_compute)
			.map_err(|e| MlError::SessionBuild(e.to_string()))?;

		session_builder = session_builder
			.with_optimization_level(inference_parameters.opt_level)
			.map_err(|e| MlError::SessionBuild(e.to_string()))?;

		if inference_parameters.gelu_approximation
		{
			session_builder = session_builder
				.with_approximate_gelu()
				.map_err(|e| MlError::SessionBuild(e.to_string()))?;
		}

		if inference_parameters.flush_to_zero
		{
			session_builder = session_builder
				.with_flush_to_zero()
				.map_err(|e| MlError::SessionBuild(e.to_string()))?;
		}

		if inference_parameters.cast_chain_elimination
		{
			session_builder = session_builder
				.with_cast_chain_elimination()
				.map_err(|e| MlError::SessionBuild(e.to_string()))?;
		}

		if let Some(inter_threads) = inference_parameters.inter_threads
		{
			session_builder = session_builder
				.with_inter_threads(inter_threads)
				.map_err(|e| MlError::SessionBuild(e.to_string()))?;
		}

		if let Some(intra_threads) = inference_parameters.intra_threads
		{
			session_builder = session_builder
				.with_intra_threads(intra_threads)
				.map_err(|e| MlError::SessionBuild(e.to_string()))?;
		}

		Ok(session_builder)
	}

	#[allow(dead_code)]
	fn compile(
		session_builder : &SessionBuilder,
		raw_model_path : &str,
		compiled_model_path : &str,
		recompile : bool,
	) -> MlResult<()>
	{
		if PathBuf::from(compiled_model_path).exists() && !recompile
		{
			return Ok(());
		}

		ModelCompiler::new(session_builder.clone())
			.map_err(MlError::Compilation)?
			.with_model_from_file(raw_model_path)
			.map_err(MlError::Compilation)?
			.with_embed_ep_context()
			.map_err(MlError::Compilation)?
			.compile_to_file(compiled_model_path)
			.map_err(MlError::Compilation)?;

		Ok(())
	}

	pub(crate) fn infer(&'_ mut self) -> MlResult<SessionOutputs<'_>>
	{
		debug!("synchronizing inference inputs");
		self.io
			.synchronize_inputs()
			.map_err(MlError::Sync)?;

		debug!("running inference session");
		self.session
			.run_binding(&self.io)
			.map_err(MlError::Inference)
	}
}
