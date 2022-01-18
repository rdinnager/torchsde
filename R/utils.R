assert_no_grad <- function(...) {
  obs <- list(...)
  for(i in seq_along(obs)) {
    ob <- obs[[i]]
    if(is_torch_tensor(ob) & ob$requires_grad) {
      rlang::abort(glue::glue("Argument {names(obs)[i]} must not require gradient."))
    }
  }
}
