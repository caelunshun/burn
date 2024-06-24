use crate::module::{Devices, Module, ModuleVisitor, ParamId};
use crate::record::{PrecisionSettings, Record};
use burn_tensor::backend::{Backend, BackendBridge};
use burn_tensor::{Bool, Int, Tensor};
use core::marker::PhantomData;

type FullPrecisionBackend<B> = <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target;

/// Implements `Module<B>` for a `Module<B::FullPrecisionBridge::Target>`.
///
/// Useful for mixed-precision models.
#[derive(Debug, Clone)]
pub struct FullPrecisionAdaptor<B: Backend, M: Module<FullPrecisionBackend<B>>> {
    inner: M,
    _marker: PhantomData<B>,
}

impl<B: Backend, M: Module<FullPrecisionBackend<B>>> Module<B> for FullPrecisionAdaptor<B, M> {
    type Record = RecordAdaptor<B, M::Record>;

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        self.inner.collect_devices(devices)
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            inner: self.inner.fork(device),
            _marker: PhantomData,
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            inner: self.inner.to_device(device),
            _marker: PhantomData,
        }
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
        struct VisitorAdaptor<'a, B, V> {
            _marker: PhantomData<B>,
            inner: &'a mut V,
        }

        impl<B, V> ModuleVisitor<FullPrecisionBackend<B>> for VisitorAdaptor<'_, B, V>
        where
            V: ModuleVisitor<B>,
            B: Backend,
        {
            fn visit_float<const D: usize>(
                &mut self,
                id: &ParamId,
                tensor: &Tensor<FullPrecisionBackend<B>, D>,
            ) {
                self.inner
                    .visit_float(id, &Tensor::from_full_precision(tensor.clone()))
            }

            fn visit_int<const D: usize>(
                &mut self,
                _id: &ParamId,
                _tensor: &Tensor<FullPrecisionBackend<B>, D, Int>,
            ) {
                todo!()
            }

            fn visit_bool<const D: usize>(
                &mut self,
                _id: &ParamId,
                _tensor: &Tensor<FullPrecisionBackend<B>, D, Bool>,
            ) {
                todo!()
            }
        }

        self.inner.visit(&mut VisitorAdaptor::<B, Visitor> {
            inner: visitor,
            _marker: PhantomData,
        })
    }

    fn map<Mapper: crate::module::ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
        struct MapperAdaptor<'a, B, M> {
            inner: &'a mut M,
            _marker: PhantomData<B>,
        }

        impl<B, M> crate::module::ModuleMapper<FullPrecisionBackend<B>> for MapperAdaptor<'_, B, M>
        where
            B: Backend,
            M: crate::module::ModuleMapper<B>,
        {
            fn map_float<const D: usize>(
                &mut self,
                id: &ParamId,
                tensor: Tensor<FullPrecisionBackend<B>, D>,
            ) -> Tensor<FullPrecisionBackend<B>, D> {
                self.inner
                    .map_float(id, Tensor::from_full_precision(tensor))
                    .into_full_precision()
            }

            fn map_int<const D: usize>(
                &mut self,
                _id: &ParamId,
                _tensor: Tensor<FullPrecisionBackend<B>, D, Int>,
            ) -> Tensor<FullPrecisionBackend<B>, D, Int> {
                todo!()
            }

            fn map_bool<const D: usize>(
                &mut self,
                _id: &ParamId,
                _tensor: Tensor<FullPrecisionBackend<B>, D, Bool>,
            ) -> Tensor<FullPrecisionBackend<B>, D, Bool> {
                todo!()
            }
        }

        Self {
            inner: self.inner.map(&mut MapperAdaptor::<B, Mapper> {
                inner: mapper,
                _marker: PhantomData,
            }),
            _marker: PhantomData,
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            inner: self.inner.load_record(record.inner),
            _marker: PhantomData,
        }
    }

    fn into_record(self) -> Self::Record {
        RecordAdaptor {
            inner: self.inner.into_record(),
            _marker: PhantomData,
        }
    }
}

pub struct RecordAdaptor<B, R> {
    inner: R,
    _marker: PhantomData<B>,
}

impl<B, R> Record<B> for RecordAdaptor<B, R>
where
    B: Backend,
    R: Record<FullPrecisionBackend<B>>,
{
    type Item<S: PrecisionSettings> = R::Item<S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.inner.into_item()
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self {
            inner: R::from_item(item, device),
            _marker: PhantomData,
        }
    }
}
