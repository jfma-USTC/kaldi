// hmm/transition-model-2D.cc

// Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
//        Johns Hopkins University (author: Guoguo Chen)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include "hmm/transition-model-2D.h"
#include "tree/context-dep.h"

namespace kaldi {

	// （2d-situation）初始化TransitionModel_2D类中的vector<Tuple> tuples_，first check OK
	void TransitionModel_2D::ComputeTuples() {
		if (IsHmm())
			ComputeTuplesIsHmm();
		//else
			//ComputeTuplesNotHmm();

		// now tuples_ is populated with all possible tuples of (phone, hmm_state, pdf, self_loop_pdf).
		std::sort(tuples_.begin(), tuples_.end());  // sort to enable reverse lookup.
		// this sorting defines the transition-ids.
	}
	void TransitionModel_2D::ComputeTuplesIsHmm() {
		const std::vector<int32> &phones = topo_.GetPhones(); // 注意，这里的phones都是按升序排序好的
		KALDI_ASSERT(!phones.empty());
		int32 max_phone = phones.back();
		phone2tuples_index_.resize(max_phone + 1, -1);
		int32 pdf = 0;
		for (size_t i = 0; i < phones.size(); i++) {
			int32 phone = phones[i];
			phone2tuples_index_[phone] = static_cast<int32>(tuples_.size());
			for (size_t hmm_state = 0; hmm_state < topo_.NumPdfClasses(phone); hmm_state++) {
				tuples_.push_back(Tuple(phone, hmm_state, pdf, pdf));
				pdf++;
			}
		}
	}
	
	// （1d-situation）初始化TransitionModel_2D类中的vector<Tuple> tuples_
	/*
	void TransitionModel_2D::ComputeTuples(const ContextDependencyInterface &ctx_dep) {
		if (IsHmm())
			ComputeTuplesIsHmm(ctx_dep);
		else
			ComputeTuplesNotHmm(ctx_dep);

		// now tuples_ is populated with all possible tuples of (phone, hmm_state, pdf, self_loop_pdf).
		std::sort(tuples_.begin(), tuples_.end());  // sort to enable reverse lookup.
		// this sorting defines the transition-ids.
	}	

	void TransitionModel_2D::ComputeTuplesIsHmm(const ContextDependencyInterface &ctx_dep) {
		const std::vector<int32> &phones = topo_.GetPhones();
		KALDI_ASSERT(!phones.empty());

		// this is the case for normal models. but not fot chain models
		std::vector<std::vector<std::pair<int32, int32> > > pdf_info;
		// num_pdf_classes表征了从phone到pdf_num的映射
		std::vector<int32> num_pdf_classes(1 + *std::max_element(phones.begin(), phones.end()), -1);
		for (size_t i = 0; i < phones.size(); i++)
			num_pdf_classes[phones[i]] = topo_.NumPdfClasses(phones[i]);
		ctx_dep.GetPdfInfo(phones, num_pdf_classes, &pdf_info);
		// pdf_info is list indexed by pdf of which (phone, pdf_class) it
		// can correspond to.

		std::map<std::pair<int32, int32>, std::vector<int32> > to_hmm_state_list;
		// to_hmm_state_list is a map from (phone, pdf_class) to the list
		// of hmm-states in the HMM for that phone that that (phone, pdf-class)
		// can correspond to.
		for (size_t i = 0; i < phones.size(); i++) {  // setting up to_hmm_state_list.
			int32 phone = phones[i];
			const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
			for (int32 j = 0; j < static_cast<int32>(entry.size()); j++) {  // for each state...
				int32 pdf_class = entry[j].forward_pdf_class;
				if (pdf_class != kNoPdf) {
					to_hmm_state_list[std::make_pair(phone, pdf_class)].push_back(j);
				}
			}
		}

		for (int32 pdf = 0; pdf < static_cast<int32>(pdf_info.size()); pdf++) {
			for (size_t j = 0; j < pdf_info[pdf].size(); j++) {
				int32 phone = pdf_info[pdf][j].first,
					pdf_class = pdf_info[pdf][j].second;
				const std::vector<int32> &state_vec = to_hmm_state_list[std::make_pair(phone, pdf_class)];
				KALDI_ASSERT(!state_vec.empty());
				// state_vec is a list of the possible HMM-states that emit this
				// pdf_class.
				for (size_t k = 0; k < state_vec.size(); k++) {
					int32 hmm_state = state_vec[k];
					tuples_.push_back(Tuple(phone, hmm_state, pdf, pdf));
				}
			}
		}
	}

	void TransitionModel_2D::ComputeTuplesNotHmm(const ContextDependencyInterface &ctx_dep) {
		const std::vector<int32> &phones = topo_.GetPhones();
		KALDI_ASSERT(!phones.empty());

		// pdf_info is a set of lists indexed by phone. Each list is indexed by
		// (pdf-class, self-loop pdf-class) of each state of that phone, and the element
		// is a list of possible (pdf, self-loop pdf) pairs that that (pdf-class, self-loop pdf-class)
		// pair generates.
		std::vector<std::vector<std::vector<std::pair<int32, int32> > > > pdf_info;
		// pdf_class_pairs is a set of lists indexed by phone. Each list stores
		// (pdf-class, self-loop pdf-class) of each state of that phone.
		std::vector<std::vector<std::pair<int32, int32> > > pdf_class_pairs;
		pdf_class_pairs.resize(1 + *std::max_element(phones.begin(), phones.end()));
		for (size_t i = 0; i < phones.size(); i++) {
			int32 phone = phones[i];
			const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
			for (int32 j = 0; j < static_cast<int32>(entry.size()); j++) {  // for each state...
				int32 forward_pdf_class = entry[j].forward_pdf_class, self_loop_pdf_class = entry[j].self_loop_pdf_class;
				if (forward_pdf_class != kNoPdf)
					pdf_class_pairs[phone].push_back(std::make_pair(forward_pdf_class, self_loop_pdf_class));
			}
		}
		ctx_dep.GetPdfInfo(phones, pdf_class_pairs, &pdf_info);

		std::vector<std::map<std::pair<int32, int32>, std::vector<int32> > > to_hmm_state_list;
		to_hmm_state_list.resize(1 + *std::max_element(phones.begin(), phones.end()));
		// to_hmm_state_list is a phone-indexed set of maps from (pdf-class, self-loop pdf_class) to the list
		// of hmm-states in the HMM for that phone that that (pdf-class, self-loop pdf-class)
		// can correspond to.
		for (size_t i = 0; i < phones.size(); i++) {  // setting up to_hmm_state_list.
			int32 phone = phones[i];
			const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
			std::map<std::pair<int32, int32>, std::vector<int32> > phone_to_hmm_state_list;
			for (int32 j = 0; j < static_cast<int32>(entry.size()); j++) {  // for each state...
				int32 forward_pdf_class = entry[j].forward_pdf_class, self_loop_pdf_class = entry[j].self_loop_pdf_class;
				if (forward_pdf_class != kNoPdf) {
					phone_to_hmm_state_list[std::make_pair(forward_pdf_class, self_loop_pdf_class)].push_back(j);
				}
			}
			to_hmm_state_list[phone] = phone_to_hmm_state_list;
		}

		for (int32 i = 0; i < phones.size(); i++) {
			int32 phone = phones[i];
			for (int32 j = 0; j < static_cast<int32>(pdf_info[phone].size()); j++) {
				int32 pdf_class = pdf_class_pairs[phone][j].first,
					self_loop_pdf_class = pdf_class_pairs[phone][j].second;
				const std::vector<int32> &state_vec =
					to_hmm_state_list[phone][std::make_pair(pdf_class, self_loop_pdf_class)];
				KALDI_ASSERT(!state_vec.empty());
				for (size_t k = 0; k < state_vec.size(); k++) {
					int32 hmm_state = state_vec[k];
					for (size_t m = 0; m < pdf_info[phone][j].size(); m++) {
						int32 pdf = pdf_info[phone][j][m].first,
							self_loop_pdf = pdf_info[phone][j][m].second;
						tuples_.push_back(Tuple(phone, hmm_state, pdf, self_loop_pdf));
					}
				}
			}
		}
	}

	*/

	//初始化TransitionModel_2D类中的state2id_ id2state_ id2pdf_id_ num_pdfs_，first check OK
	void TransitionModel_2D::ComputeDerived() {
		state2id_top_down_.resize(tuples_.size() + 2);
		state2id_left_right_.resize(tuples_.size() + 2); // indexed by transition-state, which
		// is one based, but also an entry for one past end of list. 
		// tuples_.size()为总的state数目（204×9）
		// trans_state is 1 based, tuples_ is 0 based
		// 标号为 i 的 trans_state 对应 tuples_[i - 1]

		int32 cur_transition_id_top_down = 1;
		int32 cur_transition_id_left_right = 1;
		num_pdfs_ = 0;
		// 遍历所有emitting states, state2id_top_down_[tstate]存储着tstate对应的transition_id_top_down_起始值
		// 相邻两个state2id_top_down_元素相减即为前一元素的transition_id_top_down_的数目
		for (int32 tstate = 1;
			tstate <= static_cast<int32>(tuples_.size() + 1);  // not a typo.
			tstate++) {
			state2id_top_down_[tstate] = cur_transition_id_top_down;
			state2id_left_right_[tstate] = cur_transition_id_left_right;
			if (static_cast<size_t>(tstate) <= tuples_.size()) {
				int32 phone = tuples_[tstate - 1].phone,
					hmm_state = tuples_[tstate - 1].hmm_state,
					forward_pdf = tuples_[tstate - 1].forward_pdf,
					self_loop_pdf = tuples_[tstate - 1].self_loop_pdf;
				num_pdfs_ = std::max(num_pdfs_, 1 + forward_pdf);
				num_pdfs_ = std::max(num_pdfs_, 1 + self_loop_pdf);
				const HmmTopology_2D::HmmState_2D &state = topo_.TopologyForPhone(phone)[hmm_state];
				int32 my_num_ids_top_down = static_cast<int32>(state.transitions_top_down.size());
				cur_transition_id_top_down += my_num_ids_top_down;  // # trans out of this state.
				int32 my_num_ids_left_right = static_cast<int32>(state.transitions_left_right.size());
				cur_transition_id_left_right += my_num_ids_left_right;  // # trans out of this state.
			}
		}

		id2state_top_down_.resize(cur_transition_id_top_down);   // cur_transition_id is #transition-ids+1.
		id2state_left_right_.resize(cur_transition_id_left_right); 
		id2pdf_id_top_down_.resize(cur_transition_id_top_down);
		id2pdf_id_left_right_.resize(cur_transition_id_left_right);
		for (int32 tstate = 1; tstate <= static_cast<int32>(tuples_.size()); tstate++) {
			for (int32 tid = state2id_top_down_[tstate]; tid < state2id_top_down_[tstate + 1]; tid++) {
				id2state_top_down_[tid] = tstate;
				if (IsSelfLoop_TopDown(tid))
					id2pdf_id_top_down_[tid] = tuples_[tstate - 1].self_loop_pdf;
				else
					id2pdf_id_top_down_[tid] = tuples_[tstate - 1].forward_pdf;
			}
			for (int32 tid = state2id_left_right_[tstate]; tid < state2id_left_right_[tstate + 1]; tid++) {
				id2state_left_right_[tid] = tstate;
				if (IsSelfLoop_LeftRight(tid))
					id2pdf_id_left_right_[tid] = tuples_[tstate - 1].self_loop_pdf;
				else
					id2pdf_id_left_right_[tid] = tuples_[tstate - 1].forward_pdf;
			}
		}

		// The following statements put copies a large number in the region of memory
		// past the end of the id2pdf_id_ array, while leaving the aray as it was
		// before.  The goal of this is to speed up decoding by disabling a check
		// inside TransitionIdToPdf() that the transition-id was within the correct
		// range.
		/// these statements double id2pdf_id by max_int or add 2000 max_int
		int32 num_big_numbers = std::min<int32>(2000, cur_transition_id_top_down);
		id2pdf_id_top_down_.resize(cur_transition_id_top_down + num_big_numbers,
			std::numeric_limits<int32>::max());
		id2pdf_id_top_down_.resize(cur_transition_id_top_down);

		num_big_numbers = std::min<int32>(2000, cur_transition_id_left_right);
		id2pdf_id_left_right_.resize(cur_transition_id_left_right + num_big_numbers,
			std::numeric_limits<int32>::max());
		id2pdf_id_left_right_.resize(cur_transition_id_left_right);
	}

	//初始化TransitionModel_2D类中的log_probs_ non_self_loop_log_probs_，first check OK
	void TransitionModel_2D::InitializeProbs() {
		log_probs_top_down_.Resize(NumTransitionIds_TopDown() + 1);  // one-based array, zeroth element empty.
		log_probs_left_right_.Resize(NumTransitionIds_LeftRight() + 1);  // one-based array, zeroth element empty.
		for (int32 trans_id = 1; trans_id <= NumTransitionIds_TopDown(); trans_id++) {
			int32 trans_state = id2state_top_down_[trans_id];
			int32 trans_index = trans_id - state2id_top_down_[trans_state];
			const Tuple &tuple = tuples_[trans_state - 1];
			const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(tuple.phone);
			KALDI_ASSERT(static_cast<size_t>(tuple.hmm_state) < entry.size());
			BaseFloat prob = entry[tuple.hmm_state].transitions_top_down[trans_index].second;
			if (prob <= 0.0)
				KALDI_ERR << "TransitionModel_2D::InitializeProbs, zero "
				"probability [should remove that entry in the topology]";
			if (prob > 1.0)
				KALDI_WARN << "TransitionModel_2D::InitializeProbs, prob greater than one.";
			log_probs_top_down_(trans_id) = Log(prob);
		}
		for (int32 trans_id = 1; trans_id <= NumTransitionIds_LeftRight(); trans_id++) {
			int32 trans_state = id2state_left_right_[trans_id];
			int32 trans_index = trans_id - state2id_left_right_[trans_state];
			const Tuple &tuple = tuples_[trans_state - 1];
			const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(tuple.phone);
			KALDI_ASSERT(static_cast<size_t>(tuple.hmm_state) < entry.size());
			BaseFloat prob = entry[tuple.hmm_state].transitions_left_right[trans_index].second;
			if (prob <= 0.0)
				KALDI_ERR << "TransitionModel_2D::InitializeProbs, zero "
				"probability [should remove that entry in the topology]";
			if (prob > 1.0)
				KALDI_WARN << "TransitionModel_2D::InitializeProbs, prob greater than one.";
			log_probs_left_right_(trans_id) = Log(prob);
		}
		ComputeDerivedOfProbs();
	}

	//对emitting states个数、各种对应函数做出正确性检查，first check OK
	void TransitionModel_2D::Check() const {
		KALDI_ASSERT(NumTransitionIds_TopDown() != 0 && NumTransitionIds_LeftRight() != 0 && NumTransitionStates() != 0);
		{
			int32 sum_top_down = 0;
			int32 sum_left_right = 0;
			for (int32 ts = 1; ts <= NumTransitionStates(); ts++) {
				sum_top_down += NumTransitionIndices_TopDown(ts);
				sum_left_right += NumTransitionIndices_LeftRight(ts);
			}
			KALDI_ASSERT(sum_top_down == NumTransitionIds_TopDown());
			KALDI_ASSERT(sum_left_right == NumTransitionIds_LeftRight());
		}
		for (int32 tid = 1; tid <= NumTransitionIds_TopDown(); tid++) {
			int32 tstate = TransitionIdToTransitionState_TopDown(tid),
				index = TransitionIdToTransitionIndex_TopDown(tid);
			KALDI_ASSERT(tstate > 0 && tstate <= NumTransitionStates() && index >= 0);
			KALDI_ASSERT(tid == PairToTransitionId_TopDown(tstate, index));
			int32 phone = TransitionStateToPhone(tstate),
				hmm_state = TransitionStateToHmmState(tstate),
				forward_pdf = TransitionStateToForwardPdf(tstate),
				self_loop_pdf = TransitionStateToSelfLoopPdf(tstate);
			KALDI_ASSERT(tstate == TupleToTransitionState(phone, hmm_state, forward_pdf, self_loop_pdf));
			KALDI_ASSERT(log_probs_top_down_(tid) <= 0.0 && log_probs_top_down_(tid) - log_probs_top_down_(tid) == 0.0);
			// checking finite and non-positive (and not out-of-bounds).
		}
		for (int32 tid = 1; tid <= NumTransitionIds_LeftRight(); tid++) {
			int32 tstate = TransitionIdToTransitionState_LeftRight(tid),
				index = TransitionIdToTransitionIndex_LeftRight(tid);
			KALDI_ASSERT(tstate > 0 && tstate <= NumTransitionStates() && index >= 0);
			KALDI_ASSERT(tid == PairToTransitionId_LeftRight(tstate, index));
			int32 phone = TransitionStateToPhone(tstate),
				hmm_state = TransitionStateToHmmState(tstate),
				forward_pdf = TransitionStateToForwardPdf(tstate),
				self_loop_pdf = TransitionStateToSelfLoopPdf(tstate);
			KALDI_ASSERT(tstate == TupleToTransitionState(phone, hmm_state, forward_pdf, self_loop_pdf));
			KALDI_ASSERT(log_probs_left_right_(tid) <= 0.0 && log_probs_left_right_(tid) - log_probs_left_right_(tid) == 0.0);
			// checking finite and non-positive (and not out-of-bounds).
		}
	}

	//对于每一个音素检查该音素中的每个state的前向概率和自转概率pdf_class是否相同，都相同则是HMM，first check OK
	bool TransitionModel_2D::IsHmm() const {
		const std::vector<int32> &phones = topo_.GetPhones();
		KALDI_ASSERT(!phones.empty());
		for (size_t i = 0; i < phones.size(); i++) {
			int32 phone = phones[i];
			const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(phone);
			for (int32 j = 0; j < static_cast<int32>(entry.size()); j++) {  // for each state...
				if (entry[j].forward_pdf_class != entry[j].self_loop_pdf_class)
					return false;
			}
		}
		return true;
	}

	//利用hmm_topo初始化了转移模型，包括tuples_ state2id_ id2state_ id2pdf_id_ log_probs_ non_self_loop_log_probs_ num_pdfs_
	TransitionModel_2D::TransitionModel_2D(const HmmTopology_2D &hmm_topo) : topo_(hmm_topo) {
		// First thing is to get all possible tuples.
		ComputeTuples();
		ComputeDerived();
		InitializeProbs();
		Check();
	}

	//返回由四元组指向的trans-state，first check OK
	int32 TransitionModel_2D::TupleToTransitionState(int32 phone, int32 hmm_state, int32 pdf, int32 self_loop_pdf) const {
		Tuple tuple(phone, hmm_state, pdf, self_loop_pdf);
		// Note: if this ever gets too expensive, which is unlikely, we can refactor
		// this code to sort first on pdf, and then index on pdf, so those
		// that have the same pdf are in a contiguous range.
		std::vector<Tuple>::const_iterator iter =
			std::lower_bound(tuples_.begin(), tuples_.end(), tuple);
		if (iter == tuples_.end() || !(*iter == tuple)) {
			KALDI_ERR << "TransitionModel_2D::TupleToTransitionState, tuple not found."
				<< " (incompatible tree and model?)";
		}
		// tuples_ is indexed by transition_state-1, so add one.
		return static_cast<int32>((iter - tuples_.begin())) + 1;
	}

	//返回从trans_state出发的弧的个数，first check OK
	int32 TransitionModel_2D::NumTransitionIndices_TopDown(int32 trans_state) const {
		KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
		return static_cast<int32>(state2id_top_down_[trans_state + 1] - state2id_top_down_[trans_state]);
	}
	//返回从trans_state出发的弧的个数，first check OK
	int32 TransitionModel_2D::NumTransitionIndices_LeftRight(int32 trans_state) const {
		KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
		return static_cast<int32>(state2id_left_right_[trans_state + 1] - state2id_left_right_[trans_state]);
	}

	//返回弧trans_id对应的出发state，first check OK
	int32 TransitionModel_2D::TransitionIdToTransitionState_TopDown(int32 trans_id) const {
		KALDI_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_top_down_.size());
		return id2state_top_down_[trans_id];
	}
	//返回弧trans_id对应的出发state，first check OK
	int32 TransitionModel_2D::TransitionIdToTransitionState_LeftRight(int32 trans_id) const {
		KALDI_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_left_right_.size());
		return id2state_left_right_[trans_id];
	}

	//返回trans_id在出发state的转移弧列表中的trans_index（是出发state的第几个弧），first check OK
	int32 TransitionModel_2D::TransitionIdToTransitionIndex_TopDown(int32 trans_id) const {
		KALDI_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_top_down_.size());
		return trans_id - state2id_top_down_[id2state_top_down_[trans_id]];
	}
	//返回trans_id在出发state的转移弧列表中的trans_index（是出发state的第几个弧），first check OK
	int32 TransitionModel_2D::TransitionIdToTransitionIndex_LeftRight(int32 trans_id) const {
		KALDI_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_left_right_.size());
		return trans_id - state2id_left_right_[id2state_left_right_[trans_id]];
	}

	//返回trans_state所属的音素，first check OK
	int32 TransitionModel_2D::TransitionStateToPhone(int32 trans_state) const {
		KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
		return tuples_[trans_state - 1].phone;
	}

	//返回trans_state对应tuple中的forward_pdf，first check OK
	int32 TransitionModel_2D::TransitionStateToForwardPdf(int32 trans_state) const {
		KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
		return tuples_[trans_state - 1].forward_pdf;
	}

	//返回trans_state在所属音素中对应hmm_state的forward_pdf_class，first check OK
	int32 TransitionModel_2D::TransitionStateToForwardPdfClass(
		int32 trans_state) const {
		KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
		const Tuple &t = tuples_[trans_state - 1];
		const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(t.phone);
		KALDI_ASSERT(static_cast<size_t>(t.hmm_state) < entry.size());
		return entry[t.hmm_state].forward_pdf_class;
	}

	//返回trans_state在所属音素中对应hmm_state的self_loop_pdf_class，first check OK
	int32 TransitionModel_2D::TransitionStateToSelfLoopPdfClass(
		int32 trans_state) const {
		KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
		const Tuple &t = tuples_[trans_state - 1];
		const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(t.phone);
		KALDI_ASSERT(static_cast<size_t>(t.hmm_state) < entry.size());
		return entry[t.hmm_state].self_loop_pdf_class;
	}

	//返回trans_state对应tuple中的self_loop_pdf，first check OK
	int32 TransitionModel_2D::TransitionStateToSelfLoopPdf(int32 trans_state) const {
		KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
		return tuples_[trans_state - 1].self_loop_pdf;
	}

	//返回trans_state在所在音素中的位置，0 based，first check OK
	int32 TransitionModel_2D::TransitionStateToHmmState(int32 trans_state) const {
		KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
		return tuples_[trans_state - 1].hmm_state;
	}

	//返回由trans_state,trans_index指定的trans_id，first check OK
	int32 TransitionModel_2D::PairToTransitionId_TopDown(int32 trans_state, int32 trans_index) const {
		KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
		KALDI_ASSERT(trans_index < state2id_top_down_[trans_state + 1] - state2id_top_down_[trans_state]);
		return state2id_top_down_[trans_state] + trans_index;
	}
	//返回由trans_state,trans_index指定的trans_id，first check OK
	int32 TransitionModel_2D::PairToTransitionId_LeftRight(int32 trans_state, int32 trans_index) const {
		KALDI_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
		KALDI_ASSERT(trans_index < state2id_left_right_[trans_state + 1] - state2id_left_right_[trans_state]);
		return state2id_left_right_[trans_state] + trans_index;
	}

	//返回音素的总数目（其实是返回tuple_中存储的音素列表的最大值），first check OK
	int32 TransitionModel_2D::NumPhones() const {
		int32 num_trans_state = tuples_.size();
		int32 max_phone_id = 0;
		for (int32 i = 0; i < num_trans_state; ++i) {
			if (tuples_[i].phone > max_phone_id)
				max_phone_id = tuples_[i].phone;
		}
		return max_phone_id;
	}

	//由trans_id得到state和trans_index，结合Hmm-topology-2D的entry中存储的transitions vector
	//通过判断对应位置的transitions target是否等于该entry的总size来判断trans_id是否指向最后一个state
	//注意，这里每个phone的final_state都应该是non-emitting state，first check OK
	bool TransitionModel_2D::IsFinal_TopDown(int32 trans_id) const {
		KALDI_ASSERT(static_cast<size_t>(trans_id) < id2state_top_down_.size());
		int32 trans_state = id2state_top_down_[trans_id];
		int32 trans_index = trans_id - state2id_top_down_[trans_state];
		const Tuple &tuple = tuples_[trans_state - 1];
		const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(tuple.phone);
		KALDI_ASSERT(static_cast<size_t>(tuple.hmm_state) < entry.size());
		KALDI_ASSERT(static_cast<size_t>(trans_index) <
			entry[tuple.hmm_state].transitions_top_down.size());
		// return true if the transition goes to the final state of the
		// topology entry.
		return (entry[tuple.hmm_state].transitions_top_down[trans_index].first + 1 ==
			static_cast<int32>(entry.size()));
	}
	//判断trans_id是否指向当前phone的最后一个state，first check OK
	bool TransitionModel_2D::IsFinal_LeftRight(int32 trans_id) const {
		KALDI_ASSERT(static_cast<size_t>(trans_id) < id2state_left_right_.size());
		int32 trans_state = id2state_left_right_[trans_id];
		int32 trans_index = trans_id - state2id_left_right_[trans_state];
		const Tuple &tuple = tuples_[trans_state - 1];
		const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(tuple.phone);
		KALDI_ASSERT(static_cast<size_t>(tuple.hmm_state) < entry.size());
		KALDI_ASSERT(static_cast<size_t>(trans_index) <
			entry[tuple.hmm_state].transitions_left_right.size());
		// return true if the transition goes to the final state of the
		// topology entry.
		return (entry[tuple.hmm_state].transitions_left_right[trans_index].first + 1 ==
			static_cast<int32>(entry.size()));
	}

	//返回对应于trans_state的自转弧的trans_id标号，first check OK
	int32 TransitionModel_2D::SelfLoopOf_TopDown(int32 trans_state) const {  // returns the self-loop transition-id,
		KALDI_ASSERT(static_cast<size_t>(trans_state - 1) < tuples_.size());
		const Tuple &tuple = tuples_[trans_state - 1];
		// or zero if does not exist.
		int32 phone = tuple.phone, hmm_state = tuple.hmm_state;
		const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(phone);
		KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
		for (int32 trans_index = 0;
			trans_index < static_cast<int32>(entry[hmm_state].transitions_top_down.size());
			trans_index++)
			if (entry[hmm_state].transitions_top_down[trans_index].first == hmm_state)
				return PairToTransitionId_TopDown(trans_state, trans_index);
		return 0;  // invalid transition id.
	}
	//返回对应于trans_state的自转弧的trans_id标号，first check OK
	int32 TransitionModel_2D::SelfLoopOf_LeftRight(int32 trans_state) const {  // returns the self-loop transition-id,
		KALDI_ASSERT(static_cast<size_t>(trans_state - 1) < tuples_.size());
		const Tuple &tuple = tuples_[trans_state - 1];
		// or zero if does not exist.
		int32 phone = tuple.phone, hmm_state = tuple.hmm_state;
		const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(phone);
		KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
		for (int32 trans_index = 0;
			trans_index < static_cast<int32>(entry[hmm_state].transitions_left_right.size());
			trans_index++)
			if (entry[hmm_state].transitions_left_right[trans_index].first == hmm_state)
				return PairToTransitionId_LeftRight(trans_state, trans_index);
		return 0;  // invalid transition id. 这意味着trans_state不存在自转弧
	}

	//初始化TransitionModel_2D类中的non_self_loop_log_probs_, both directions，first check OK
	void TransitionModel_2D::ComputeDerivedOfProbs() {
		non_self_loop_log_probs_top_down_.Resize(NumTransitionStates() + 1);  // this array indexed
		non_self_loop_log_probs_left_right_.Resize(NumTransitionStates() + 1);  // this array indexed
		//  by transition-state with nothing in zeroth element.
		for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
			int32 tid_top_down = SelfLoopOf_TopDown(tstate);
			int32 tid_left_right = SelfLoopOf_LeftRight(tstate);
			if (tid_top_down == 0) {  // no self-loop
				non_self_loop_log_probs_top_down_(tstate) = 0.0;  // log(1.0)
			}
			else {
				BaseFloat self_loop_prob = Exp(GetTransitionLogProb_TopDown(tid_top_down));
				BaseFloat non_self_loop_prob = 1.0 - self_loop_prob;
				if (non_self_loop_prob <= 0.0) {
					KALDI_WARN << "ComputeDerivedOfProbs(): non-self-loop prob top2down is " << non_self_loop_prob;
					non_self_loop_prob = 1.0e-10;  // just so we can continue...
				}
				non_self_loop_log_probs_top_down_(tstate) = Log(non_self_loop_prob);  // will be negative.
			}
			if (tid_left_right == 0) {  // no self-loop
				non_self_loop_log_probs_left_right_(tstate) = 0.0;  // log(1.0)
			}
			else {
				BaseFloat self_loop_prob = Exp(GetTransitionLogProb_TopDown(tid_left_right));
				BaseFloat non_self_loop_prob = 1.0 - self_loop_prob;
				if (non_self_loop_prob <= 0.0) {
					KALDI_WARN << "ComputeDerivedOfProbs(): non-self-loop prob left2right is " << non_self_loop_prob;
					non_self_loop_prob = 1.0e-10;  // just so we can continue...
				}
				non_self_loop_log_probs_left_right_(tstate) = Log(non_self_loop_prob);  // will be negative.
			}
		}
	}

	//标准读取函数，读取.mdl文件中<TransitionModel_2D>...</TransitionModel_2D>部分，first check OK
	void TransitionModel_2D::Read(std::istream &is, bool binary) {
		ExpectToken(is, binary, "<TransitionModel_2D>");
		topo_.Read(is, binary);
		std::string token;
		ReadToken(is, binary, &token);
		int32 size;
		ReadBasicType(is, binary, &size);
		tuples_.resize(size);
		for (int32 i = 0; i < size; i++) {
			ReadBasicType(is, binary, &(tuples_[i].phone));
			ReadBasicType(is, binary, &(tuples_[i].hmm_state));
			ReadBasicType(is, binary, &(tuples_[i].forward_pdf));
			if (token == "<Tuples>")
				ReadBasicType(is, binary, &(tuples_[i].self_loop_pdf));
			else if (token == "<Triples>")
				tuples_[i].self_loop_pdf = tuples_[i].forward_pdf;
		}
		ReadToken(is, binary, &token);
		KALDI_ASSERT(token == "</Triples>" || token == "</Tuples>");
		ComputeDerived();
		ExpectToken(is, binary, "<LogProbs_TopDown>");
		log_probs_top_down_.Read(is, binary);
		ExpectToken(is, binary, "</LogProbs_TopDown>");
		ExpectToken(is, binary, "<LogProbs_LeftRight>");
		log_probs_left_right_.Read(is, binary);
		ExpectToken(is, binary, "</LogProbs_LeftRight>");
		ExpectToken(is, binary, "</TransitionModel_2D>");
		ComputeDerivedOfProbs();
		Check();
	}

	//标准写入函数，写入.mdl文件中<TransitionModel_2D>...</TransitionModel_2D>部分，first check OK
	void TransitionModel_2D::Write(std::ostream &os, bool binary) const {
		bool is_hmm = IsHmm(); // always true in 2D
		WriteToken(os, binary, "<TransitionModel_2D>");
		if (!binary) os << "\n";
		topo_.Write(os, binary);
		if (is_hmm)
			WriteToken(os, binary, "<Triples>");
		else
			WriteToken(os, binary, "<Tuples>");
		WriteBasicType(os, binary, static_cast<int32>(tuples_.size()));
		if (!binary) os << "\n";
		for (int32 i = 0; i < static_cast<int32> (tuples_.size()); i++) {
			WriteBasicType(os, binary, tuples_[i].phone);
			WriteBasicType(os, binary, tuples_[i].hmm_state);
			WriteBasicType(os, binary, tuples_[i].self_loop_pdf);
			if (!is_hmm)
			  WriteBasicType(os, binary, tuples_[i].self_loop_pdf);
			if (!binary) os << "\n";
		}
		if (is_hmm)
			WriteToken(os, binary, "</Triples>");
		else
			WriteToken(os, binary, "</Tuples>");
		if (!binary) os << "\n";
		WriteToken(os, binary, "<LogProbs_TopDown>");
		if (!binary) os << "\n";
		log_probs_top_down_.Write(os, binary);
		WriteToken(os, binary, "</LogProbs_TopDown>");
		if (!binary) os << "\n";
		WriteToken(os, binary, "<LogProbs_LeftRight>");
		if (!binary) os << "\n";
		log_probs_left_right_.Write(os, binary);
		WriteToken(os, binary, "</LogProbs_LeftRight>");
		if (!binary) os << "\n";
		WriteToken(os, binary, "</TransitionModel_2D>");
		if (!binary) os << "\n";
	}

	//返回trans_id指定的转移概率(不是对数概率)，first check OK
	BaseFloat TransitionModel_2D::GetTransitionProb_TopDown(int32 trans_id) const {
		return Exp(log_probs_top_down_(trans_id));
	}
	//返回trans_id指定的转移概率(不是对数概率)，first check OK
	BaseFloat TransitionModel_2D::GetTransitionProb_LeftRight(int32 trans_id) const {
		return Exp(log_probs_left_right_(trans_id));
	}

	//返回trans_id指定的对数概率，first check OK
	BaseFloat TransitionModel_2D::GetTransitionLogProb_TopDown(int32 trans_id) const {
		return log_probs_top_down_(trans_id);
	}
	//返回trans_id指定的对数概率，first check OK
	BaseFloat TransitionModel_2D::GetTransitionLogProb_LeftRight(int32 trans_id) const {
		return log_probs_left_right_(trans_id);
	}

	//返回trans_state的非自转对数概率，first check OK
	BaseFloat TransitionModel_2D::GetNonSelfLoopLogProb_TopDown(int32 trans_state) const {
		KALDI_ASSERT(trans_state != 0);
		return non_self_loop_log_probs_top_down_(trans_state);
	}
	//返回trans_state的非自转对数概率，first check OK
	BaseFloat TransitionModel_2D::GetNonSelfLoopLogProb_LeftRight(int32 trans_state) const {
		KALDI_ASSERT(trans_state != 0);
		return non_self_loop_log_probs_left_right_(trans_state);
	}

	//返回trans_id占所在trans_state概率比例的对数值（log(trans_id÷none_self_loop)），first check OK
	BaseFloat TransitionModel_2D::GetTransitionLogProbIgnoringSelfLoops_TopDown(int32 trans_id) const {
		KALDI_ASSERT(trans_id != 0);
		KALDI_PARANOID_ASSERT(!IsSelfLoop_TopDown(trans_id));
		return log_probs_top_down_(trans_id) - GetNonSelfLoopLogProb_TopDown(TransitionIdToTransitionState_TopDown(trans_id));
	}
	//返回trans_id占所在trans_state概率比例的对数值（log(trans_id÷none_self_loop)），first check OK
	BaseFloat TransitionModel_2D::GetTransitionLogProbIgnoringSelfLoops_LeftRight(int32 trans_id) const {
		KALDI_ASSERT(trans_id != 0);
		KALDI_PARANOID_ASSERT(!IsSelfLoop_LeftRight(trans_id));
		return log_probs_left_right_(trans_id) - GetNonSelfLoopLogProb_LeftRight(TransitionIdToTransitionState_LeftRight(trans_id));
	}

	// stats are counts/weights, indexed by transition-id.，first check OK
	void TransitionModel_2D::MleUpdate_TopDown(const Vector<double> &stats,
		const MleTransitionUpdateConfig &cfg,
		BaseFloat *objf_impr_out,
		BaseFloat *count_out) {
		if (cfg.share_for_pdfs) {
			MleUpdateShared_TopDown(stats, cfg, objf_impr_out, count_out);
			return;
		}
		BaseFloat count_sum = 0.0, objf_impr_sum = 0.0;
		int32 num_skipped = 0, num_floored = 0;
		KALDI_ASSERT(stats.Dim() == NumTransitionIds_TopDown() + 1);
		for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
			int32 n = NumTransitionIndices_TopDown(tstate); // NumTransitionIndices返回指定trans_state对应的转移弧数目
			KALDI_ASSERT(n >= 1);
			if (n > 1) {  // no point updating if only one transition...
				Vector<double> counts(n);
				for (int32 tidx = 0; tidx < n; tidx++) {
					int32 tid = PairToTransitionId_TopDown(tstate, tidx);
					counts(tidx) = stats(tid); // 使用counts向量记录某个trans_state对应的所有转移弧出现的weighted次数（由stats（trans_acc统计量）提供）
				}
				double tstate_tot = counts.Sum(); // tstate_tot表示某个trans_state对应的弧出现的总次数
				count_sum += tstate_tot; // 统计所有弧出现的总次数
				if (tstate_tot < cfg.mincount) { num_skipped++; } // mincount默认为5，如果某个state对应的转移弧总数少于5则不更新该state
				else {
					Vector<BaseFloat> old_probs(n), new_probs(n);
					for (int32 tidx = 0; tidx < n; tidx++) { // tidx表示当前state可选的trans_id标号
						int32 tid = PairToTransitionId_TopDown(tstate, tidx);
						old_probs(tidx) = new_probs(tidx) = GetTransitionProb_TopDown(tid); // 将旧TM对应的该弧的转移概率赋给新旧两个概率值
					}
					for (int32 tidx = 0; tidx < n; tidx++)
						new_probs(tidx) = counts(tidx) / tstate_tot; // 简单的除法更新转移概率
					for (int32 i = 0; i < 3; i++) {  // keep flooring+renormalizing for 3 times..
						new_probs.Scale(1.0 / new_probs.Sum());
						for (int32 tidx = 0; tidx < n; tidx++)
							new_probs(tidx) = std::max(new_probs(tidx), cfg.floor); // 防止出现过小的转移概率
					}
					// Compute objf change
					for (int32 tidx = 0; tidx < n; tidx++) {
						if (new_probs(tidx) == cfg.floor) num_floored++;
						double objf_change = counts(tidx) * (Log(new_probs(tidx))
							- Log(old_probs(tidx)));
						objf_impr_sum += objf_change;
					}
					// Commit updated values.
					for (int32 tidx = 0; tidx < n; tidx++) {
						int32 tid = PairToTransitionId_TopDown(tstate, tidx);
						log_probs_top_down_(tid) = Log(new_probs(tidx)); // log_probs_存放的是转移概率的对数值
						if (log_probs_top_down_(tid) - log_probs_top_down_(tid) != 0.0)
							KALDI_ERR << "Log probs TopDown is inf or NaN: error in update or bad stats?";
					}
				}
			}
		}
		KALDI_LOG << "TransitionModel_2D::MleUpdate_TopDown, objf change is "
			<< (objf_impr_sum / count_sum) << " per frame over " << count_sum
			<< " frames. ";
		KALDI_LOG << num_floored << " probabilities floored, " << num_skipped
			<< " out of " << NumTransitionStates() << " transition-states "
			"skipped due to insuffient data (it is normal to have some skipped.)";
		if (objf_impr_out) *objf_impr_out = objf_impr_sum;
		if (count_out) *count_out = count_sum;
		ComputeDerivedOfProbs();
	}
	// stats are counts/weights, indexed by transition-id.，first check OK
	void TransitionModel_2D::MleUpdate_LeftRight(const Vector<double> &stats,
		const MleTransitionUpdateConfig &cfg,
		BaseFloat *objf_impr_out,
		BaseFloat *count_out) {
		if (cfg.share_for_pdfs) {
			MleUpdateShared_LeftRight(stats, cfg, objf_impr_out, count_out);
			return;
		}
		BaseFloat count_sum = 0.0, objf_impr_sum = 0.0;
		int32 num_skipped = 0, num_floored = 0;
		KALDI_ASSERT(stats.Dim() == NumTransitionIds_LeftRight() + 1);
		for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
			int32 n = NumTransitionIndices_LeftRight(tstate); // NumTransitionIndices返回指定trans_state对应的转移弧数目
			KALDI_ASSERT(n >= 1);
			if (n > 1) {  // no point updating if only one transition...
				Vector<double> counts(n);
				for (int32 tidx = 0; tidx < n; tidx++) {
					int32 tid = PairToTransitionId_LeftRight(tstate, tidx);
					counts(tidx) = stats(tid); // 使用counts向量记录某个trans_state对应的所有转移弧出现的次数（由stats（trans_acc统计量）提供）
				}
				double tstate_tot = counts.Sum(); // tstate_tot表示某个trans_state对应的弧出现的总次数
				count_sum += tstate_tot; // 统计所有弧出现的总次数
				if (tstate_tot < cfg.mincount) { num_skipped++; } // mincount默认为5，如果某个state对应的转移弧总数少于5则不更新该state
				else {
					Vector<BaseFloat> old_probs(n), new_probs(n);
					for (int32 tidx = 0; tidx < n; tidx++) { // tidx表示当前state可选的trans_id标号
						int32 tid = PairToTransitionId_LeftRight(tstate, tidx);
						old_probs(tidx) = new_probs(tidx) = GetTransitionProb_LeftRight(tid); // 将旧TM对应的该弧的转移概率赋给新旧两个概率值
					}
					for (int32 tidx = 0; tidx < n; tidx++)
						new_probs(tidx) = counts(tidx) / tstate_tot; // 简单的除法更新转移概率
					for (int32 i = 0; i < 3; i++) {  // keep flooring+renormalizing for 3 times..
						new_probs.Scale(1.0 / new_probs.Sum());
						for (int32 tidx = 0; tidx < n; tidx++)
							new_probs(tidx) = std::max(new_probs(tidx), cfg.floor); // 防止出现过小的转移概率
					}
					// Compute objf change
					for (int32 tidx = 0; tidx < n; tidx++) {
						if (new_probs(tidx) == cfg.floor) num_floored++;
						double objf_change = counts(tidx) * (Log(new_probs(tidx))
							- Log(old_probs(tidx)));
						objf_impr_sum += objf_change;
					}
					// Commit updated values.
					for (int32 tidx = 0; tidx < n; tidx++) {
						int32 tid = PairToTransitionId_LeftRight(tstate, tidx);
						log_probs_left_right_(tid) = Log(new_probs(tidx)); // log_probs_存放的是转移概率的对数值
						if (log_probs_left_right_(tid) - log_probs_left_right_(tid) != 0.0)
							KALDI_ERR << "Log probs is inf or NaN: error in update or bad stats?";
					}
				}
			}
		}
		KALDI_LOG << "TransitionModel_2D::MleUpdate_LeftRight, objf change is "
			<< (objf_impr_sum / count_sum) << " per frame over " << count_sum
			<< " frames. ";
		KALDI_LOG << num_floored << " probabilities floored, " << num_skipped
			<< " out of " << NumTransitionStates() << " transition-states "
			"skipped due to insuffient data (it is normal to have some skipped.)";
		if (objf_impr_out) *objf_impr_out = objf_impr_sum;
		if (count_out) *count_out = count_sum;
		ComputeDerivedOfProbs();
	}

	// stats are counts/weights, indexed by transition-id.，first check OK
	void TransitionModel_2D::MapUpdate_TopDown(const Vector<double> &stats,
		const MapTransitionUpdateConfig &cfg,
		BaseFloat *objf_impr_out,
		BaseFloat *count_out) {
		KALDI_ASSERT(cfg.tau > 0.0);
		if (cfg.share_for_pdfs) {
			MapUpdateShared_TopDown(stats, cfg, objf_impr_out, count_out);
			return;
		}
		BaseFloat count_sum = 0.0, objf_impr_sum = 0.0;
		KALDI_ASSERT(stats.Dim() == NumTransitionIds_TopDown() + 1);
		for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
			int32 n = NumTransitionIndices_TopDown(tstate);
			KALDI_ASSERT(n >= 1);
			if (n > 1) {  // no point updating if only one transition...
				Vector<double> counts(n);
				for (int32 tidx = 0; tidx < n; tidx++) {
					int32 tid = PairToTransitionId_TopDown(tstate, tidx);
					counts(tidx) = stats(tid);
				}
				double tstate_tot = counts.Sum();
				count_sum += tstate_tot;
				Vector<BaseFloat> old_probs(n), new_probs(n);
				for (int32 tidx = 0; tidx < n; tidx++) {
					int32 tid = PairToTransitionId_TopDown(tstate, tidx);
					old_probs(tidx) = new_probs(tidx) = GetTransitionProb_TopDown(tid);
				}
				for (int32 tidx = 0; tidx < n; tidx++)
					new_probs(tidx) = (counts(tidx) + cfg.tau * old_probs(tidx)) /
					(cfg.tau + tstate_tot);
				// Compute objf change
				for (int32 tidx = 0; tidx < n; tidx++) {
					double objf_change = counts(tidx) * (Log(new_probs(tidx))
						- Log(old_probs(tidx)));
					objf_impr_sum += objf_change;
				}
				// Commit updated values.
				for (int32 tidx = 0; tidx < n; tidx++) {
					int32 tid = PairToTransitionId_TopDown(tstate, tidx);
					log_probs_top_down_(tid) = Log(new_probs(tidx));
					if (log_probs_top_down_(tid) - log_probs_top_down_(tid) != 0.0)
						KALDI_ERR << "Log probs is inf or NaN: error in update or bad stats?";
				}
			}
		}
		KALDI_LOG << "TransitionModel_2D::MapUpdate_TopDown, objf change is " << (objf_impr_sum / count_sum)
			<< " per frame over " << count_sum
			<< " frames.";
		if (objf_impr_out) *objf_impr_out = objf_impr_sum;
		if (count_out) *count_out = count_sum;
		ComputeDerivedOfProbs();
	}
	// stats are counts/weights, indexed by transition-id.，first check OK
	void TransitionModel_2D::MapUpdate_LeftRight(const Vector<double> &stats,
		const MapTransitionUpdateConfig &cfg,
		BaseFloat *objf_impr_out,
		BaseFloat *count_out) {
		KALDI_ASSERT(cfg.tau > 0.0);
		if (cfg.share_for_pdfs) {
			MapUpdateShared_LeftRight(stats, cfg, objf_impr_out, count_out);
			return;
		}
		BaseFloat count_sum = 0.0, objf_impr_sum = 0.0;
		KALDI_ASSERT(stats.Dim() == NumTransitionIds_LeftRight() + 1);
		for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
			int32 n = NumTransitionIndices_LeftRight(tstate);
			KALDI_ASSERT(n >= 1);
			if (n > 1) {  // no point updating if only one transition...
				Vector<double> counts(n);
				for (int32 tidx = 0; tidx < n; tidx++) {
					int32 tid = PairToTransitionId_LeftRight(tstate, tidx);
					counts(tidx) = stats(tid);
				}
				double tstate_tot = counts.Sum();
				count_sum += tstate_tot;
				Vector<BaseFloat> old_probs(n), new_probs(n);
				for (int32 tidx = 0; tidx < n; tidx++) {
					int32 tid = PairToTransitionId_LeftRight(tstate, tidx);
					old_probs(tidx) = new_probs(tidx) = GetTransitionProb_LeftRight(tid);
				}
				for (int32 tidx = 0; tidx < n; tidx++)
					new_probs(tidx) = (counts(tidx) + cfg.tau * old_probs(tidx)) /
					(cfg.tau + tstate_tot);
				// Compute objf change
				for (int32 tidx = 0; tidx < n; tidx++) {
					double objf_change = counts(tidx) * (Log(new_probs(tidx))
						- Log(old_probs(tidx)));
					objf_impr_sum += objf_change;
				}
				// Commit updated values.
				for (int32 tidx = 0; tidx < n; tidx++) {
					int32 tid = PairToTransitionId_LeftRight(tstate, tidx);
					log_probs_left_right_(tid) = Log(new_probs(tidx));
					if (log_probs_left_right_(tid) - log_probs_left_right_(tid) != 0.0)
						KALDI_ERR << "Log probs is inf or NaN: error in update or bad stats?";
				}
			}
		}
		KALDI_LOG << "TransitionModel_2D::MapUpdate_LeftRight, objf change is " << (objf_impr_sum / count_sum)
			<< " per frame over " << count_sum
			<< " frames.";
		if (objf_impr_out) *objf_impr_out = objf_impr_sum;
		if (count_out) *count_out = count_sum;
		ComputeDerivedOfProbs();
	}

	/// This version of the Update() function is for if the user specifies
	/// --share-for-pdfs=true.  We share the transitions for all states that
	/// share the same pdf.，first check OK
	void TransitionModel_2D::MleUpdateShared_TopDown(const Vector<double> &stats,
		const MleTransitionUpdateConfig &cfg,
		BaseFloat *objf_impr_out,
		BaseFloat *count_out) {
		KALDI_ASSERT(cfg.share_for_pdfs);

		BaseFloat count_sum = 0.0, objf_impr_sum = 0.0;
		int32 num_skipped = 0, num_floored = 0;
		KALDI_ASSERT(stats.Dim() == NumTransitionIds_TopDown() + 1);
		std::map<int32, std::set<int32> > pdf_to_tstate;
		//pdf_to_state中存储每个pdf(tuples_中存储的与trans_state一一对应的pdf)包含的trans_state个数

		for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
			int32 pdf = TransitionStateToForwardPdf(tstate);
			pdf_to_tstate[pdf].insert(tstate);
			if (!IsHmm()) {
				pdf = TransitionStateToSelfLoopPdf(tstate);
				pdf_to_tstate[pdf].insert(tstate);
			}
		}
		std::map<int32, std::set<int32> >::iterator map_iter;
		for (map_iter = pdf_to_tstate.begin();
			map_iter != pdf_to_tstate.end();
			++map_iter) {
			// map_iter->first is pdf-id... not needed.
			const std::set<int32> &tstates = map_iter->second;
			KALDI_ASSERT(!tstates.empty());
			int32 one_tstate = *(tstates.begin());
			int32 n = NumTransitionIndices_TopDown(one_tstate);
			KALDI_ASSERT(n >= 1);
			if (n > 1) { // Only update if >1 transition...
				Vector<double> counts(n);
				for (std::set<int32>::const_iterator iter = tstates.begin();
					iter != tstates.end();
					++iter) {
					int32 tstate = *iter;
					if (NumTransitionIndices_TopDown(tstate) != n)
						KALDI_ERR << "Mismatch in #transition indices: you cannot "
						"use the --share-for-pdfs option with this topology "
						"and sharing scheme.";
					for (int32 tidx = 0; tidx < n; tidx++) {
						int32 tid = PairToTransitionId_TopDown(tstate, tidx);
						counts(tidx) += stats(tid);
					}
				}
				double pdf_tot = counts.Sum();
				count_sum += pdf_tot;
				if (pdf_tot < cfg.mincount) { num_skipped++; }
				else {
					// Note: when calculating objf improvement, we
					// assume we previously had the same tying scheme so
					// we can get the params from one_tstate and they're valid
					// for all.
					Vector<BaseFloat> old_probs(n), new_probs(n);
					for (int32 tidx = 0; tidx < n; tidx++) {
						int32 tid = PairToTransitionId_TopDown(one_tstate, tidx);
						old_probs(tidx) = new_probs(tidx) = GetTransitionProb_TopDown(tid);
					}
					for (int32 tidx = 0; tidx < n; tidx++)
						new_probs(tidx) = counts(tidx) / pdf_tot;
					for (int32 i = 0; i < 3; i++) {  // keep flooring+renormalizing for 3 times..
						new_probs.Scale(1.0 / new_probs.Sum());
						for (int32 tidx = 0; tidx < n; tidx++)
							new_probs(tidx) = std::max(new_probs(tidx), cfg.floor);
					}
					// Compute objf change
					for (int32 tidx = 0; tidx < n; tidx++) {
						if (new_probs(tidx) == cfg.floor) num_floored++;
						double objf_change = counts(tidx) * (Log(new_probs(tidx))
							- Log(old_probs(tidx)));
						objf_impr_sum += objf_change;
					}
					// Commit updated values.
					for (std::set<int32>::const_iterator iter = tstates.begin();
						iter != tstates.end();
						++iter) {
						int32 tstate = *iter;
						for (int32 tidx = 0; tidx < n; tidx++) {
							int32 tid = PairToTransitionId_TopDown(tstate, tidx);
							log_probs_top_down_(tid) = Log(new_probs(tidx));
							if (log_probs_top_down_(tid) - log_probs_top_down_(tid) != 0.0)
								KALDI_ERR << "Log probs is inf or NaN: error in update or bad stats?";
						}
					}
				}
			}
		}
		KALDI_LOG << "TransitionModel_2D::MleUpdateShared_TopDown, objf change is " << (objf_impr_sum / count_sum)
			<< " per frame over " << count_sum << " frames; "
			<< num_floored << " probabilities floored, "
			<< num_skipped << " pdf-ids skipped due to insuffient data.";
		if (objf_impr_out) *objf_impr_out = objf_impr_sum;
		if (count_out) *count_out = count_sum;
		ComputeDerivedOfProbs();
	}
	//share the transitions for all states that share the same pdf.，first check OK
	void TransitionModel_2D::MleUpdateShared_LeftRight(const Vector<double> &stats,
		const MleTransitionUpdateConfig &cfg,
		BaseFloat *objf_impr_out,
		BaseFloat *count_out) {
		KALDI_ASSERT(cfg.share_for_pdfs);

		BaseFloat count_sum = 0.0, objf_impr_sum = 0.0;
		int32 num_skipped = 0, num_floored = 0;
		KALDI_ASSERT(stats.Dim() == NumTransitionIds_LeftRight() + 1);
		std::map<int32, std::set<int32> > pdf_to_tstate;
		//pdf_to_state中存储每个pdf(tuples_中存储的与trans_state一一对应的pdf)包含的trans_state个数

		for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
			int32 pdf = TransitionStateToForwardPdf(tstate);
			pdf_to_tstate[pdf].insert(tstate);
			if (!IsHmm()) {
				pdf = TransitionStateToSelfLoopPdf(tstate);
				pdf_to_tstate[pdf].insert(tstate);
			}
		}
		std::map<int32, std::set<int32> >::iterator map_iter;
		for (map_iter = pdf_to_tstate.begin();
			map_iter != pdf_to_tstate.end();
			++map_iter) {
			// map_iter->first is pdf-id... not needed.
			const std::set<int32> &tstates = map_iter->second;
			KALDI_ASSERT(!tstates.empty());
			int32 one_tstate = *(tstates.begin());
			int32 n = NumTransitionIndices_LeftRight(one_tstate);
			KALDI_ASSERT(n >= 1);
			if (n > 1) { // Only update if >1 transition...
				Vector<double> counts(n);
				for (std::set<int32>::const_iterator iter = tstates.begin();
					iter != tstates.end();
					++iter) {
					int32 tstate = *iter;
					if (NumTransitionIndices_LeftRight(tstate) != n)
						KALDI_ERR << "Mismatch in #transition indices: you cannot "
						"use the --share-for-pdfs option with this topology "
						"and sharing scheme.";
					for (int32 tidx = 0; tidx < n; tidx++) {
						int32 tid = PairToTransitionId_LeftRight(tstate, tidx);
						counts(tidx) += stats(tid);
					}
				}
				double pdf_tot = counts.Sum();
				count_sum += pdf_tot;
				if (pdf_tot < cfg.mincount) { num_skipped++; }
				else {
					// Note: when calculating objf improvement, we
					// assume we previously had the same tying scheme so
					// we can get the params from one_tstate and they're valid
					// for all.
					Vector<BaseFloat> old_probs(n), new_probs(n);
					for (int32 tidx = 0; tidx < n; tidx++) {
						int32 tid = PairToTransitionId_LeftRight(one_tstate, tidx);
						old_probs(tidx) = new_probs(tidx) = GetTransitionProb_LeftRight(tid);
					}
					for (int32 tidx = 0; tidx < n; tidx++)
						new_probs(tidx) = counts(tidx) / pdf_tot;
					for (int32 i = 0; i < 3; i++) {  // keep flooring+renormalizing for 3 times..
						new_probs.Scale(1.0 / new_probs.Sum());
						for (int32 tidx = 0; tidx < n; tidx++)
							new_probs(tidx) = std::max(new_probs(tidx), cfg.floor);
					}
					// Compute objf change
					for (int32 tidx = 0; tidx < n; tidx++) {
						if (new_probs(tidx) == cfg.floor) num_floored++;
						double objf_change = counts(tidx) * (Log(new_probs(tidx))
							- Log(old_probs(tidx)));
						objf_impr_sum += objf_change;
					}
					// Commit updated values.
					for (std::set<int32>::const_iterator iter = tstates.begin();
						iter != tstates.end();
						++iter) {
						int32 tstate = *iter;
						for (int32 tidx = 0; tidx < n; tidx++) {
							int32 tid = PairToTransitionId_LeftRight(tstate, tidx);
							log_probs_left_right_(tid) = Log(new_probs(tidx));
							if (log_probs_left_right_(tid) - log_probs_left_right_(tid) != 0.0)
								KALDI_ERR << "Log probs is inf or NaN: error in update or bad stats?";
						}
					}
				}
			}
		}
		KALDI_LOG << "TransitionModel_2D::MleUpdateShared_LeftRight, objf change is " << (objf_impr_sum / count_sum)
			<< " per frame over " << count_sum << " frames; "
			<< num_floored << " probabilities floored, "
			<< num_skipped << " pdf-ids skipped due to insuffient data.";
		if (objf_impr_out) *objf_impr_out = objf_impr_sum;
		if (count_out) *count_out = count_sum;
		ComputeDerivedOfProbs();
	}


	/// This version of the MapUpdate() function is for if the user specifies
	/// --share-for-pdfs=true.  We share the transitions for all states that
	/// share the same pdf. first check OK
	void TransitionModel_2D::MapUpdateShared_TopDown(const Vector<double> &stats,
		const MapTransitionUpdateConfig &cfg,
		BaseFloat *objf_impr_out,
		BaseFloat *count_out) {
		KALDI_ASSERT(cfg.share_for_pdfs);

		BaseFloat count_sum = 0.0, objf_impr_sum = 0.0;
		KALDI_ASSERT(stats.Dim() == NumTransitionIds_TopDown() + 1);
		std::map<int32, std::set<int32> > pdf_to_tstate;

		for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
			int32 pdf = TransitionStateToForwardPdf(tstate);
			pdf_to_tstate[pdf].insert(tstate);
			if (!IsHmm()) {
				pdf = TransitionStateToSelfLoopPdf(tstate);
				pdf_to_tstate[pdf].insert(tstate);
			}
		}
		std::map<int32, std::set<int32> >::iterator map_iter;
		for (map_iter = pdf_to_tstate.begin();
			map_iter != pdf_to_tstate.end();
			++map_iter) {
			// map_iter->first is pdf-id... not needed.
			const std::set<int32> &tstates = map_iter->second;
			KALDI_ASSERT(!tstates.empty());
			int32 one_tstate = *(tstates.begin());
			int32 n = NumTransitionIndices_TopDown(one_tstate);
			KALDI_ASSERT(n >= 1);
			if (n > 1) { // Only update if >1 transition...
				Vector<double> counts(n);
				for (std::set<int32>::const_iterator iter = tstates.begin();
					iter != tstates.end();
					++iter) {
					int32 tstate = *iter;
					if (NumTransitionIndices_TopDown(tstate) != n)
						KALDI_ERR << "Mismatch in #transition indices: you cannot "
						"use the --share-for-pdfs option with this topology "
						"and sharing scheme.";
					for (int32 tidx = 0; tidx < n; tidx++) {
						int32 tid = PairToTransitionId_TopDown(tstate, tidx);
						counts(tidx) += stats(tid);
					}
				}
				double pdf_tot = counts.Sum();
				count_sum += pdf_tot;

				// Note: when calculating objf improvement, we
				// assume we previously had the same tying scheme so
				// we can get the params from one_tstate and they're valid
				// for all.
				Vector<BaseFloat> old_probs(n), new_probs(n);
				for (int32 tidx = 0; tidx < n; tidx++) {
					int32 tid = PairToTransitionId_TopDown(one_tstate, tidx);
					old_probs(tidx) = new_probs(tidx) = GetTransitionProb_TopDown(tid);
				}
				for (int32 tidx = 0; tidx < n; tidx++)
					new_probs(tidx) = (counts(tidx) + old_probs(tidx) * cfg.tau) /
					(pdf_tot + cfg.tau);
				// Compute objf change
				for (int32 tidx = 0; tidx < n; tidx++) {
					double objf_change = counts(tidx) * (Log(new_probs(tidx))
						- Log(old_probs(tidx)));
					objf_impr_sum += objf_change;
				}
				// Commit updated values.
				for (std::set<int32>::const_iterator iter = tstates.begin();
					iter != tstates.end();
					++iter) {
					int32 tstate = *iter;
					for (int32 tidx = 0; tidx < n; tidx++) {
						int32 tid = PairToTransitionId_TopDown(tstate, tidx);
						log_probs_top_down_(tid) = Log(new_probs(tidx));
						if (log_probs_top_down_(tid) - log_probs_top_down_(tid) != 0.0)
							KALDI_ERR << "Log probs is inf or NaN: error in update or bad stats?";
					}
				}
			}
		}
		KALDI_LOG << "TransitionModel_2D::MapUpdateShared_TopDown, objf change is " << (objf_impr_sum / count_sum)
			<< " per frame over " << count_sum
			<< " frames.";
		if (objf_impr_out) *objf_impr_out = objf_impr_sum;
		if (count_out) *count_out = count_sum;
		ComputeDerivedOfProbs();
	}
	//We share the transitions for all states that share the same pdf. first check OK
	void TransitionModel_2D::MapUpdateShared_LeftRight(const Vector<double> &stats,
		const MapTransitionUpdateConfig &cfg,
		BaseFloat *objf_impr_out,
		BaseFloat *count_out) {
		KALDI_ASSERT(cfg.share_for_pdfs);

		BaseFloat count_sum = 0.0, objf_impr_sum = 0.0;
		KALDI_ASSERT(stats.Dim() == NumTransitionIds_LeftRight() + 1);
		std::map<int32, std::set<int32> > pdf_to_tstate;

		for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
			int32 pdf = TransitionStateToForwardPdf(tstate);
			pdf_to_tstate[pdf].insert(tstate);
			if (!IsHmm()) {
				pdf = TransitionStateToSelfLoopPdf(tstate);
				pdf_to_tstate[pdf].insert(tstate);
			}
		}
		std::map<int32, std::set<int32> >::iterator map_iter;
		for (map_iter = pdf_to_tstate.begin();
			map_iter != pdf_to_tstate.end();
			++map_iter) {
			// map_iter->first is pdf-id... not needed.
			const std::set<int32> &tstates = map_iter->second;
			KALDI_ASSERT(!tstates.empty());
			int32 one_tstate = *(tstates.begin());
			int32 n = NumTransitionIndices_LeftRight(one_tstate);
			KALDI_ASSERT(n >= 1);
			if (n > 1) { // Only update if >1 transition...
				Vector<double> counts(n);
				for (std::set<int32>::const_iterator iter = tstates.begin();
					iter != tstates.end();
					++iter) {
					int32 tstate = *iter;
					if (NumTransitionIndices_LeftRight(tstate) != n)
						KALDI_ERR << "Mismatch in #transition indices: you cannot "
						"use the --share-for-pdfs option with this topology "
						"and sharing scheme.";
					for (int32 tidx = 0; tidx < n; tidx++) {
						int32 tid = PairToTransitionId_LeftRight(tstate, tidx);
						counts(tidx) += stats(tid);
					}
				}
				double pdf_tot = counts.Sum();
				count_sum += pdf_tot;

				// Note: when calculating objf improvement, we
				// assume we previously had the same tying scheme so
				// we can get the params from one_tstate and they're valid
				// for all.
				Vector<BaseFloat> old_probs(n), new_probs(n);
				for (int32 tidx = 0; tidx < n; tidx++) {
					int32 tid = PairToTransitionId_LeftRight(one_tstate, tidx);
					old_probs(tidx) = new_probs(tidx) = GetTransitionProb_LeftRight(tid);
				}
				for (int32 tidx = 0; tidx < n; tidx++)
					new_probs(tidx) = (counts(tidx) + old_probs(tidx) * cfg.tau) /
					(pdf_tot + cfg.tau);
				// Compute objf change
				for (int32 tidx = 0; tidx < n; tidx++) {
					double objf_change = counts(tidx) * (Log(new_probs(tidx))
						- Log(old_probs(tidx)));
					objf_impr_sum += objf_change;
				}
				// Commit updated values.
				for (std::set<int32>::const_iterator iter = tstates.begin();
					iter != tstates.end();
					++iter) {
					int32 tstate = *iter;
					for (int32 tidx = 0; tidx < n; tidx++) {
						int32 tid = PairToTransitionId_LeftRight(tstate, tidx);
						log_probs_left_right_(tid) = Log(new_probs(tidx));
						if (log_probs_left_right_(tid) - log_probs_left_right_(tid) != 0.0)
							KALDI_ERR << "Log probs is inf or NaN: error in update or bad stats?";
					}
				}
			}
		}
		KALDI_LOG << "TransitionModel_2D::MapUpdateShared_LeftRight, objf change is " << (objf_impr_sum / count_sum)
			<< " per frame over " << count_sum
			<< " frames.";
		if (objf_impr_out) *objf_impr_out = objf_impr_sum;
		if (count_out) *count_out = count_sum;
		ComputeDerivedOfProbs();
	}

	//返回trans_id所在的phone，first check OK
	int32 TransitionModel_2D::TransitionIdToPhone_TopDown(int32 trans_id) const {
		KALDI_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_top_down_.size());
		int32 trans_state = id2state_top_down_[trans_id];
		return tuples_[trans_state - 1].phone;
	}
	//返回trans_id所在的phone，first check OK
	int32 TransitionModel_2D::TransitionIdToPhone_LeftRight(int32 trans_id) const {
		KALDI_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_left_right_.size());
		int32 trans_state = id2state_left_right_[trans_id];
		return tuples_[trans_state - 1].phone;
	}

	//返回trans_id对应的pdf_class，若是自转弧则返回该状态self_loop_pdf_class，若不是则返回forward_pdf_class
	//注意，返回的pdf_class为所在phone内部的标号（一个entry内），不是总的标号，first check OK
	int32 TransitionModel_2D::TransitionIdToPdfClass_TopDown(int32 trans_id) const {
		KALDI_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_top_down_.size());
		int32 trans_state = id2state_top_down_[trans_id];

		const Tuple &t = tuples_[trans_state - 1];
		const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(t.phone);
		KALDI_ASSERT(static_cast<size_t>(t.hmm_state) < entry.size());
		if (IsSelfLoop_TopDown(trans_id))
			return entry[t.hmm_state].self_loop_pdf_class;
		else
			return entry[t.hmm_state].forward_pdf_class;
	}
	//注意，返回的pdf_class为所在phone内部的标号（一个entry内），不是总的标号，first check OK
	int32 TransitionModel_2D::TransitionIdToPdfClass_LeftRight(int32 trans_id) const {
		KALDI_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_left_right_.size());
		int32 trans_state = id2state_left_right_[trans_id];

		const Tuple &t = tuples_[trans_state - 1];
		const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(t.phone);
		KALDI_ASSERT(static_cast<size_t>(t.hmm_state) < entry.size());
		if (IsSelfLoop_LeftRight(trans_id))
			return entry[t.hmm_state].self_loop_pdf_class;
		else
			return entry[t.hmm_state].forward_pdf_class;
	}

	//返回trans_id所在state在phone内的标号（HmmState），0 based，first check OK
	int32 TransitionModel_2D::TransitionIdToHmmState_TopDown(int32 trans_id) const {
		KALDI_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_top_down_.size());
		int32 trans_state = id2state_top_down_[trans_id];
		const Tuple &t = tuples_[trans_state - 1];
		return t.hmm_state;
	}
	//返回trans_id所在state在phone内的标号（HmmState），0 based，first check OK
	int32 TransitionModel_2D::TransitionIdToHmmState_LeftRight(int32 trans_id) const {
		KALDI_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_left_right_.size());
		int32 trans_state = id2state_left_right_[trans_id];
		const Tuple &t = tuples_[trans_state - 1];
		return t.hmm_state;
	}

	//输出show-transitions命令的结果，打印转移结构以及转移概率，first check OK
	void TransitionModel_2D::Print(std::ostream &os,
		const std::vector<std::string> &phone_names,
		const Vector<double> *occs) {
		if (occs != NULL)
			KALDI_ASSERT(occs->Dim() == NumPdfs());
		bool is_hmm = IsHmm();
		for (int32 tstate = 1; tstate <= NumTransitionStates(); tstate++) {
			const Tuple &tuple = tuples_[tstate - 1];
			KALDI_ASSERT(static_cast<size_t>(tuple.phone) < phone_names.size());
			std::string phone_name = phone_names[tuple.phone];

			os << "Transition-state " << tstate << ": phone = " << phone_name
				<< " hmm-state = " << tuple.hmm_state;
			if (is_hmm)
				os << " pdf = " << tuple.forward_pdf << '\n';
			else
				os << " forward-pdf = " << tuple.forward_pdf << " self-loop-pdf = "
				<< tuple.self_loop_pdf << '\n';
			for (int32 tidx = 0; tidx < NumTransitionIndices_TopDown(tstate); tidx++) {
				int32 tid = PairToTransitionId_TopDown(tstate, tidx);
				BaseFloat p = GetTransitionProb_TopDown(tid);
				os << " Transition-id-TopDown = " << tid << " p = " << p;
				if (occs != NULL) {
					if (IsSelfLoop_TopDown(tid))
						os << " count of pdf = " << (*occs)(tuple.self_loop_pdf);
					else
						os << " count of pdf = " << (*occs)(tuple.forward_pdf);
				}
				// now describe what it's a transition to.
				if (IsSelfLoop_TopDown(tid)) os << " [self-loop]\n";
				else {
					int32 hmm_state = tuple.hmm_state;
					const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(tuple.phone);
					KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
					int32 next_hmm_state = entry[hmm_state].transitions_top_down[tidx].first;
					KALDI_ASSERT(next_hmm_state != hmm_state);
					os << " [" << hmm_state << " -> " << next_hmm_state << "]\n";
				}
			}
			for (int32 tidx = 0; tidx < NumTransitionIndices_LeftRight(tstate); tidx++) {
				int32 tid = PairToTransitionId_LeftRight(tstate, tidx);
				BaseFloat p = GetTransitionProb_LeftRight(tid);
				os << " Transition-id-LeftRight = " << tid << " p = " << p;
				if (occs != NULL) {
					if (IsSelfLoop_LeftRight(tid))
						os << " count of pdf = " << (*occs)(tuple.self_loop_pdf);
					else
						os << " count of pdf = " << (*occs)(tuple.forward_pdf);
				}
				// now describe what it's a transition to.
				if (IsSelfLoop_LeftRight(tid)) os << " [self-loop]\n";
				else {
					int32 hmm_state = tuple.hmm_state;
					const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(tuple.phone);
					KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
					int32 next_hmm_state = entry[hmm_state].transitions_left_right[tidx].first;
					KALDI_ASSERT(next_hmm_state != hmm_state);
					os << " [" << hmm_state << " -> " << next_hmm_state << "]\n";
				}
			}
		}
	}

	//first check OK
	bool GetPdfsForPhones(const TransitionModel_2D &trans_model,
		const std::vector<int32> &phones,
		std::vector<int32> *pdfs) {
		KALDI_ASSERT(IsSortedAndUniq(phones));
		KALDI_ASSERT(pdfs != NULL);
		pdfs->clear();
		for (int32 tstate = 1; tstate <= trans_model.NumTransitionStates(); tstate++) {
			if (std::binary_search(phones.begin(), phones.end(),
				trans_model.TransitionStateToPhone(tstate))) {
				pdfs->push_back(trans_model.TransitionStateToForwardPdf(tstate));
				pdfs->push_back(trans_model.TransitionStateToSelfLoopPdf(tstate));
			}
		}
		SortAndUniq(pdfs);

		for (int32 tstate = 1; tstate <= trans_model.NumTransitionStates(); tstate++)
			if ((std::binary_search(pdfs->begin(), pdfs->end(),
				trans_model.TransitionStateToForwardPdf(tstate)) ||
				std::binary_search(pdfs->begin(), pdfs->end(),
					trans_model.TransitionStateToSelfLoopPdf(tstate)))
				&& !std::binary_search(phones.begin(), phones.end(),
					trans_model.TransitionStateToPhone(tstate)))
				return false;
		return true;
	}
	//first check OK
	bool GetPhonesForPdfs(const TransitionModel_2D &trans_model,
		const std::vector<int32> &pdfs,
		std::vector<int32> *phones) {
		KALDI_ASSERT(IsSortedAndUniq(pdfs));
		KALDI_ASSERT(phones != NULL);
		phones->clear();
		for (int32 tstate = 1; tstate <= trans_model.NumTransitionStates(); tstate++) {
			if (std::binary_search(pdfs.begin(), pdfs.end(),
				trans_model.TransitionStateToForwardPdf(tstate)) ||
				std::binary_search(pdfs.begin(), pdfs.end(),
					trans_model.TransitionStateToSelfLoopPdf(tstate)))
				phones->push_back(trans_model.TransitionStateToPhone(tstate));
		}
		SortAndUniq(phones);

		for (int32 tstate = 1; tstate <= trans_model.NumTransitionStates(); tstate++)
			if (std::binary_search(phones->begin(), phones->end(),
				trans_model.TransitionStateToPhone(tstate))
				&& !(std::binary_search(pdfs.begin(), pdfs.end(),
					trans_model.TransitionStateToForwardPdf(tstate)) &&
					std::binary_search(pdfs.begin(), pdfs.end(),
						trans_model.TransitionStateToSelfLoopPdf(tstate))))
				return false;
		return true;
	}

	//返回两个TransitionModel实例是否一致（结构上的一致，转移概率上不要求），first check OK
	bool TransitionModel_2D::Compatible(const TransitionModel_2D &other) const {
		return (topo_ == other.topo_ && tuples_ == other.tuples_ &&
			state2id_top_down_ == other.state2id_top_down_ && 
			id2state_top_down_ == other.id2state_top_down_ &&
			state2id_left_right_ == other.state2id_left_right_ &&
			id2state_left_right_ == other.id2state_left_right_ &&
			num_pdfs_ == other.num_pdfs_);
	}

	//判断trans_id指定的转移弧是否是自转弧，first check OK
	bool TransitionModel_2D::IsSelfLoop_TopDown(int32 trans_id) const {
		KALDI_ASSERT(static_cast<size_t>(trans_id) < id2state_top_down_.size());
		int32 trans_state = id2state_top_down_[trans_id];
		int32 trans_index = trans_id - state2id_top_down_[trans_state];
		const Tuple &tuple = tuples_[trans_state - 1];
		int32 phone = tuple.phone, hmm_state = tuple.hmm_state;
		const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(phone);
		KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
		return (static_cast<size_t>(trans_index) < entry[hmm_state].transitions_top_down.size()
			&& entry[hmm_state].transitions_top_down[trans_index].first == hmm_state);
	}
	//判断trans_id指定的转移弧是否是自转弧，first check OK
	bool TransitionModel_2D::IsSelfLoop_LeftRight(int32 trans_id) const {
		KALDI_ASSERT(static_cast<size_t>(trans_id) < id2state_left_right_.size());
		int32 trans_state = id2state_left_right_[trans_id];
		int32 trans_index = trans_id - state2id_left_right_[trans_state];
		const Tuple &tuple = tuples_[trans_state - 1];
		int32 phone = tuple.phone, hmm_state = tuple.hmm_state;
		const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(phone);
		KALDI_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
		return (static_cast<size_t>(trans_index) < entry[hmm_state].transitions_left_right.size()
			&& entry[hmm_state].transitions_left_right[trans_index].first == hmm_state);
	}

	// Phone should be in Topology_2D phones_ list, hmm_state should be 0-based
	// tuple_index_ is 0-based, trans-state is 1-based.
	int32 TransitionModel_2D::PairToState(int32 phone, int32 hmm_state) const {
		if (phone >= phone2tuples_index_.size()) 
			KALDI_ERR << "Phone "<< phone <<" exceed the range of index_vector.";
		if (phone2tuples_index_[phone] == -1) 
			KALDI_ERR << "Phone " << phone << " didn't contained in index_vector.";
		return phone2tuples_index_[phone] + 1 + hmm_state;
	}

	// 返回从this_state指向next_state的转移弧trans_id，若不存在这样的转移弧则返回-1
	// 一般情况this_state和next_state都应该是某个trans_state,next_state为0意味着下一状态应该是结束态；
	int32 TransitionModel_2D::StatePairToTransitionId_TopDown(int32 this_state, int32 next_state) const {
		KALDI_ASSERT(static_cast<size_t>(this_state) <= tuples_.size());//防止无效的this_state的出现
		int32 this_phone = tuples_[this_state - 1].phone;//记录this_state属于哪一个音素
		int32 this_state_hmm_state = tuples_[this_state - 1].hmm_state;//找到this_state在该音素中的hmm_state
		int32 next_state_hmm_state;
		//记录该音素的topology_entry，其中包含了HmmState_2D 的vector列表，最后一项为nonemitting state
		//例如若this_phone为三状态，则entry.size()为4，hmm_state为0、1、2、3(nonemitting)
		const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(this_phone);
		if (next_state == 0) //若next_state为零意味着下一状态应该是结束态
			next_state_hmm_state = entry.size() - 1;
		else {
			KALDI_ASSERT(static_cast<size_t>(next_state) <= tuples_.size());
			KALDI_ASSERT(this_phone == tuples_[next_state - 1].phone);//确保两个state处于同一个音素中
			next_state_hmm_state = tuples_[next_state - 1].hmm_state;//找到next_state在该音素中的hmm_state
		}
		KALDI_ASSERT(this_state_hmm_state < entry.size());
		int32 next_state_trans_index = -1;
		// 遍历entry中属于this_state的水平转移弧列表，检查next_state是否在其中，若在则返回trans_index
		for (size_t i = 0; i < entry[this_state_hmm_state].transitions_top_down.size(); i++)
			if (next_state_hmm_state == entry[this_state_hmm_state].transitions_top_down[i].first)
				next_state_trans_index = static_cast<int32>(i);
		if (next_state_trans_index != -1)
			return state2id_top_down_[this_state] + next_state_trans_index;
		else
			return -1;
	}
	// 返回从this_state指向next_state的转移弧trans_id，若不存在这样的转移弧则返回-1
	// 一般情况this_state和next_state都应该是某个trans_state,next_state为0意味着下一状态应该是结束态；
	int32 TransitionModel_2D::StatePairToTransitionId_LeftRight(int32 this_state, int32 next_state) const {
		KALDI_ASSERT(static_cast<size_t>(this_state) <= tuples_.size());//防止无效的this_state的出现
		int32 this_phone = tuples_[this_state - 1].phone;//记录this_state属于哪一个音素
		int32 this_state_hmm_state = tuples_[this_state - 1].hmm_state;//找到this_state在该音素中的hmm_state
		int32 next_state_hmm_state;
		//记录该音素的topology_entry，其中包含了HmmState_2D 的vector列表，最后一项为nonemitting state
		//例如若this_phone为三状态，则entry.size()为4，hmm_state为0、1、2、3(nonemitting)
		const HmmTopology_2D::TopologyEntry_2D &entry = topo_.TopologyForPhone(this_phone);
		if (next_state == 0) //若next_state为零意味着下一状态应该是结束态
			next_state_hmm_state = entry.size() - 1;
		else {
			KALDI_ASSERT(static_cast<size_t>(next_state) <= tuples_.size());
			KALDI_ASSERT(this_phone == tuples_[next_state - 1].phone);//确保两个state处于同一个音素中
			next_state_hmm_state = tuples_[next_state - 1].hmm_state;//找到next_state在该音素中的hmm_state
		}
		KALDI_ASSERT(this_state_hmm_state < entry.size());
		int32 next_state_trans_index = -1;
		// 遍历entry中属于this_state的水平转移弧列表，检查next_state是否在其中，若在则返回trans_index
		for (size_t i = 0; i < entry[this_state_hmm_state].transitions_left_right.size(); i++)
			if (next_state_hmm_state == entry[this_state_hmm_state].transitions_left_right[i].first)
				next_state_trans_index = static_cast<int32>(i);
		if (next_state_trans_index != -1)
			return state2id_left_right_[this_state] + next_state_trans_index;
		else
			return -1;
	}

} // End namespace kaldi
